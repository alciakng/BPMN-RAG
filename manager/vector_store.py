# vector_store.py
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import streamlit as st

from common.logger import Logger

LOGGER = Logger.get_logger("manager.vector_store")


class _MemoryVectorBackend:
    """In-memory fallback when FAISS/Pinecone is unavailable."""

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        # session_id -> list of (query, answer, embedding, timestamp)
        self.store: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, session_id: str, query: str, answer: str, embedding: List[float]) -> None:
        """Add a Q&A pair with embedding."""
        if session_id not in self.store:
            self.store[session_id] = []

        self.store[session_id].append({
            "query": query,
            "answer": answer,
            "embedding": np.array(embedding, dtype=np.float32),
            "timestamp": int(time.time())
        })

    def search(self, session_id: str, query_embedding: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        """Search for similar Q&A pairs using cosine similarity."""
        if session_id not in self.store or not self.store[session_id]:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        # Calculate cosine similarity
        similarities = []
        for item in self.store[session_id]:
            emb = item["embedding"]
            emb_norm = np.linalg.norm(emb)

            if emb_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, emb) / (query_norm * emb_norm)

            similarities.append((sim, item))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        results = []
        for sim, item in similarities[:top_k]:
            results.append({
                "query": item["query"],
                "answer": item["answer"],
                "similarity": float(sim),
                "timestamp": item["timestamp"]
            })

        return results

    def clear(self, session_id: str) -> None:
        """Clear all Q&A pairs for a session."""
        if session_id in self.store:
            del self.store[session_id]


class FaissStore:
    """
    FAISS-based vector store (with in-memory fallback) to manage Q&A history with embeddings.

    Stores (query, answer) pairs per session_id with vector embeddings for similarity search.
    Uses Pinecone or local FAISS for vector storage.
    """

    def __init__(self, llm_client=None):
        """
        Initialize FAISS vector store.

        Args:
            llm_client: LLM client for generating embeddings (optional, can be set later)
        """
        print("[PINECONE][INIT] FaissStore.__init__ called")  # DEBUG
        self.backend = None
        self.backend_type = None  # 'pinecone', 'faiss', or 'memory'
        self.llm_client = llm_client
        self.embedding_dim = 1536  # OpenAI embedding dimension

        try:
            # Try to initialize Pinecone
            url = st.secrets.get('PINECONE_URL', None)
            api_key = st.secrets.get('PINECONE_API_KEY', None)

            print(f"[PINECONE][INIT] PINECONE_URL={url if url else 'NOT SET'} PINECONE_API_KEY={'***' + api_key[-4:] if api_key else 'NOT SET'}")  # DEBUG
            LOGGER.info("[PINECONE][INIT] PINECONE_URL=%s PINECONE_API_KEY=%s",
                       url if url else "NOT SET",
                       "***" + api_key[-4:] if api_key else "NOT SET")

            if url and api_key:
                try:
                    LOGGER.info("[PINECONE][INIT] Attempting to connect to Pinecone...")
                    import pinecone
                    from pinecone import Pinecone, ServerlessSpec

                    # Initialize Pinecone client
                    pc = Pinecone(api_key=api_key)
                    LOGGER.info("[PINECONE][INIT] Pinecone client initialized successfully")

                    # Extract index name from URL or use default
                    # URL format: https://INDEX_NAME-PROJECT_ID.svc.ENVIRONMENT.pinecone.io
                    # Extract the index name from the host
                    index_name = "qa-history"
                    if url:
                        # Remove protocol if present
                        clean_url = url.replace("https://", "").replace("http://", "")
                        # Get the host part (before any path)
                        host = clean_url.split("/")[0]
                        # Extract index name (first part before the dash and project ID)
                        # Format: bpmnrag-vqjsq16.svc... -> bpmnrag
                        if "-" in host:
                            index_name = host.split("-")[0]
                        else:
                            index_name = host.split(".")[0]

                    LOGGER.info("[PINECONE][INIT] Extracted index name from URL: %s (from %s)", index_name, url)

                    # Check if index exists, create if not
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    LOGGER.info("[PINECONE][INIT] Existing indexes: %s", existing_indexes)

                    if index_name not in existing_indexes:
                        LOGGER.info("[PINECONE][INIT] Index not found, creating index: %s", index_name)
                        pc.create_index(
                            name=index_name,
                            dimension=self.embedding_dim,
                            metric="cosine",
                            spec=ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            )
                        )
                        LOGGER.info("[PINECONE][INIT] Created Pinecone index: %s", index_name)
                    else:
                        LOGGER.info("[PINECONE][INIT] Index already exists: %s", index_name)

                    self.backend = pc.Index(index_name)
                    self.backend_type = 'pinecone'
                    LOGGER.info("[PINECONE][INIT] Successfully connected to Pinecone index: %s", index_name)
                    return

                except Exception as e:
                    LOGGER.exception("[PINECONE][INIT] Pinecone initialization failed: %s", e)

            # Try local FAISS
            LOGGER.info("[PINECONE][INIT] Pinecone not available, trying local FAISS...")
            try:
                import faiss

                # Create a simple FAISS index (FlatIP for cosine similarity)
                self.backend = faiss.IndexFlatIP(self.embedding_dim)
                # Store metadata separately (FAISS only stores vectors)
                self.metadata: Dict[str, List[Dict[str, Any]]] = {}
                self.backend_type = 'faiss'
                LOGGER.info("[PINECONE][INIT] Using local FAISS index")
                return

            except Exception as e:
                LOGGER.warning("[PINECONE][INIT] Local FAISS initialization failed: %s", e)

        except Exception as e:
            LOGGER.warning("[PINECONE][INIT] Vector store unavailable, falling back to memory. err=%s", e)

        # Fallback to in-memory
        self.backend = _MemoryVectorBackend(embedding_dim=self.embedding_dim)
        self.backend_type = 'memory'
        LOGGER.info("[PINECONE][INIT] Using in-memory vector backend")

    def set_llm_client(self, llm_client):
        """Set LLM client for generating embeddings."""
        self.llm_client = llm_client

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using LLM client."""
        if not self.llm_client:
            LOGGER.warning("[FAISS] LLM client not set, cannot generate embedding")
            return None

        try:
            # Use OpenAI's embedding API
            import openai
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            LOGGER.exception("[FAISS][EMBEDDING][ERROR] %s", e)
            return None

    def add_qa_pair(
        self,
        session_id: str,
        query: str,
        answer: str,
        model_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Add a Q&A pair to the vector store with embedding.

        Args:
            session_id: Session identifier
            query: User query
            answer: Assistant answer
            model_keys: List of model keys to associate with this Q&A pair

        Returns:
            True if successful, False otherwise
        """
        try:
            LOGGER.info("[PINECONE][ADD] Starting add_qa_pair session=%s model_keys=%s backend_type=%s",
                       session_id, model_keys, self.backend_type)

            # Generate embedding for the query
            embedding = self._get_embedding(query)
            if not embedding:
                LOGGER.warning("[PINECONE][ADD] Cannot add Q&A pair without embedding")
                return False

            LOGGER.info("[PINECONE][ADD] Generated embedding successfully, length=%d", len(embedding))

            # If no model_keys provided, use empty list
            model_keys = model_keys or []

            # Store for each model_key
            success_count = 0
            for model_key in model_keys:
                LOGGER.info("[PINECONE][ADD] Processing model_key=%s", model_key)
                try:
                    # Store based on backend type
                    if isinstance(self.backend, _MemoryVectorBackend):
                        # For memory backend, use combined key
                        combined_key = f"{session_id}_{model_key}"
                        LOGGER.info("[PINECONE][ADD] Using memory backend, combined_key=%s", combined_key)
                        self.backend.add(combined_key, query, answer, embedding)

                    elif hasattr(self.backend, 'upsert'):  # Pinecone
                        # Create unique ID with model_key
                        vector_id = f"{session_id}_{model_key}_{int(time.time())}"
                        LOGGER.info("[PINECONE][ADD] Using Pinecone backend, vector_id=%s", vector_id)

                        upsert_data = [{
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "session_id": session_id,
                                "model_key": model_key,
                                "query": query,
                                "answer": answer,
                                "timestamp": int(time.time())
                            }
                        }]

                        LOGGER.info("[PINECONE][ADD] Upserting to Pinecone: vector_id=%s metadata_keys=%s",
                                   vector_id, list(upsert_data[0]["metadata"].keys()))

                        self.backend.upsert(vectors=upsert_data)

                        LOGGER.info("[PINECONE][ADD] Successfully upserted to Pinecone")

                    else:  # Local FAISS
                        import faiss
                        LOGGER.info("[PINECONE][ADD] Using local FAISS backend")
                        # Normalize for cosine similarity
                        emb_array = np.array([embedding], dtype=np.float32)
                        faiss.normalize_L2(emb_array)
                        self.backend.add(emb_array)

                        # Store metadata with model_key
                        combined_key = f"{session_id}_{model_key}"
                        if combined_key not in self.metadata:
                            self.metadata[combined_key] = []
                        self.metadata[combined_key].append({
                            "query": query,
                            "answer": answer,
                            "timestamp": int(time.time())
                        })
                        LOGGER.info("[PINECONE][ADD] Added to FAISS metadata, combined_key=%s", combined_key)

                    success_count += 1
                    LOGGER.info(
                        "[PINECONE][ADD] Added Q&A pair session=%s model_key=%s query_len=%d answer_len=%d",
                        session_id, model_key, len(query), len(answer)
                    )

                except Exception as me:
                    LOGGER.exception("[PINECONE][ADD][MODEL=%s][ERROR] %s", model_key, me)

            LOGGER.info("[PINECONE][ADD] Completed add_qa_pair: success_count=%d total_models=%d",
                       success_count, len(model_keys))
            return success_count > 0

        except Exception as e:
            LOGGER.exception("[PINECONE][ADD][ERROR] %s", e)
            return False

    def search_similar(
        self,
        session_id: str,
        query: str,
        model_keys: Optional[List[str]] = None,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search for similar Q&A pairs using vector similarity.

        Args:
            session_id: Session identifier
            query: Current query to find similar past queries
            model_keys: List of model keys to filter search results
            top_k: Number of results to return

        Returns:
            List of similar Q&A pairs with format:
            [
                {
                    "query": "past query",
                    "answer": "past answer",
                    "similarity": 0.95,
                    "timestamp": 1234567890
                },
                ...
            ]
        """
        try:
            LOGGER.info("[PINECONE][SEARCH] Starting search_similar session=%s model_keys=%s top_k=%d backend_type=%s",
                       session_id, model_keys, top_k, self.backend_type)

            # Generate embedding for the query
            embedding = self._get_embedding(query)
            if not embedding:
                LOGGER.warning("[PINECONE][SEARCH] Cannot search without embedding")
                return []

            LOGGER.info("[PINECONE][SEARCH] Generated embedding successfully, length=%d", len(embedding))

            # If no model_keys provided, use empty list
            model_keys = model_keys or []

            # Search based on backend type
            if isinstance(self.backend, _MemoryVectorBackend):
                LOGGER.info("[PINECONE][SEARCH] Using memory backend")
                # Search across all model_keys and aggregate results
                all_results = []
                for model_key in model_keys:
                    combined_key = f"{session_id}_{model_key}"
                    results = self.backend.search(combined_key, embedding, top_k)
                    all_results.extend(results)

                # Sort by similarity and return top_k
                all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                results = all_results[:top_k]

            elif hasattr(self.backend, 'query'):  # Pinecone
                LOGGER.info("[PINECONE][SEARCH] Using Pinecone backend")
                # Build filter for session_id and model_keys
                if model_keys:
                    # Filter by session_id AND model_key in list
                    filter_dict = {
                        "$and": [
                            {"session_id": {"$eq": session_id}},
                            {"model_key": {"$in": model_keys}}
                        ]
                    }
                else:
                    # Only filter by session_id
                    filter_dict = {"session_id": {"$eq": session_id}}

                LOGGER.info("[PINECONE][SEARCH] Query filter: %s", filter_dict)

                response = self.backend.query(
                    vector=embedding,
                    top_k=top_k,
                    filter=filter_dict,
                    include_metadata=True
                )

                LOGGER.info("[PINECONE][SEARCH] Pinecone query completed")

                results = []
                matches = getattr(response, 'matches', [])
                LOGGER.info("[PINECONE][SEARCH] Found %d matches from Pinecone", len(matches))

                for match in matches:
                    metadata = getattr(match, 'metadata', {})
                    score = getattr(match, 'score', 0.0)
                    results.append({
                        "query": metadata.get("query", ""),
                        "answer": metadata.get("answer", ""),
                        "similarity": float(score),
                        "timestamp": metadata.get("timestamp", 0)
                    })
                    LOGGER.info("[PINECONE][SEARCH] Match: score=%.4f query_len=%d",
                               score, len(metadata.get("query", "")))

            else:  # Local FAISS
                LOGGER.info("[PINECONE][SEARCH] Using local FAISS backend")
                import faiss
                # Normalize query embedding
                emb_array = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(emb_array)

                # Search across all model_keys and aggregate results
                all_results = []
                for model_key in model_keys:
                    combined_key = f"{session_id}_{model_key}"
                    session_metadata = self.metadata.get(combined_key, [])

                    LOGGER.info("[PINECONE][SEARCH] FAISS search for combined_key=%s metadata_count=%d",
                               combined_key, len(session_metadata))

                    if not session_metadata:
                        continue

                    # Search in FAISS
                    distances, indices = self.backend.search(emb_array, min(top_k, len(session_metadata)))

                    # Filter by session+model and construct results
                    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                        if idx >= 0 and idx < len(session_metadata):
                            item = session_metadata[idx]
                            all_results.append({
                                "query": item["query"],
                                "answer": item["answer"],
                                "similarity": float(dist),
                                "timestamp": item["timestamp"]
                            })
                            LOGGER.info("[PINECONE][SEARCH] FAISS match: dist=%.4f idx=%d", dist, idx)

                # Sort by similarity and return top_k
                all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                results = all_results[:top_k]

            LOGGER.info(
                "[PINECONE][SEARCH] Search completed session=%s model_keys=%s query_len=%d results=%d",
                session_id, model_keys, len(query), len(results)
            )
            return results

        except Exception as e:
            LOGGER.exception("[PINECONE][SEARCH][ERROR] %s", e)
            return []

    def clear_session(self, session_id: str) -> None:
        """
        Clear all Q&A pairs for a session.

        Args:
            session_id: Session identifier
        """
        try:
            if isinstance(self.backend, _MemoryVectorBackend):
                self.backend.clear(session_id)

            elif hasattr(self.backend, 'delete'):  # Pinecone
                # Pinecone doesn't support bulk delete by metadata easily
                # Would need to fetch all IDs first, then delete
                LOGGER.warning("[FAISS] Pinecone session clear not fully implemented")

            else:  # Local FAISS
                # FAISS doesn't support deletion easily
                # Clear metadata only
                if session_id in self.metadata:
                    del self.metadata[session_id]

            LOGGER.info("[FAISS] Cleared session=%s", session_id)

        except Exception as e:
            LOGGER.exception("[FAISS][CLEAR][ERROR] %s", e)
