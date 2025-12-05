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
        self.backend = None
        self.llm_client = llm_client
        self.embedding_dim = 1536  # OpenAI embedding dimension

        try:
            # Try to initialize Pinecone
            url = st.secrets.get('FAISS_URL', None)
            api_key = st.secrets.get('FAISS_API_KEY', None)

            if url and api_key:
                try:
                    import pinecone
                    from pinecone import Pinecone, ServerlessSpec

                    # Initialize Pinecone client
                    pc = Pinecone(api_key=api_key)

                    # Extract index name from URL or use default
                    index_name = "qa-history"
                    if "/" in url:
                        index_name = url.split("/")[-1]

                    # Check if index exists, create if not
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        pc.create_index(
                            name=index_name,
                            dimension=self.embedding_dim,
                            metric="cosine",
                            spec=ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            )
                        )
                        LOGGER.info("[FAISS] Created Pinecone index: %s", index_name)

                    self.backend = pc.Index(index_name)
                    LOGGER.info("[FAISS] Connected to Pinecone: %s", index_name)
                    return

                except Exception as e:
                    LOGGER.warning("[FAISS] Pinecone initialization failed: %s", e)

            # Try local FAISS
            try:
                import faiss

                # Create a simple FAISS index (FlatIP for cosine similarity)
                self.backend = faiss.IndexFlatIP(self.embedding_dim)
                # Store metadata separately (FAISS only stores vectors)
                self.metadata: Dict[str, List[Dict[str, Any]]] = {}
                LOGGER.info("[FAISS] Using local FAISS index")
                return

            except Exception as e:
                LOGGER.warning("[FAISS] Local FAISS initialization failed: %s", e)

        except Exception as e:
            LOGGER.warning("[FAISS] Vector store unavailable, falling back to memory. err=%s", e)

        # Fallback to in-memory
        self.backend = _MemoryVectorBackend(embedding_dim=self.embedding_dim)
        LOGGER.info("[FAISS] Using in-memory vector backend")

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
        answer: str
    ) -> bool:
        """
        Add a Q&A pair to the vector store with embedding.

        Args:
            session_id: Session identifier
            query: User query
            answer: Assistant answer

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for the query
            embedding = self._get_embedding(query)
            if not embedding:
                LOGGER.warning("[FAISS] Cannot add Q&A pair without embedding")
                return False

            # Store based on backend type
            if isinstance(self.backend, _MemoryVectorBackend):
                self.backend.add(session_id, query, answer, embedding)

            elif hasattr(self.backend, 'upsert'):  # Pinecone
                # Create unique ID
                vector_id = f"{session_id}_{int(time.time())}"
                self.backend.upsert(
                    vectors=[(
                        vector_id,
                        embedding,
                        {
                            "session_id": session_id,
                            "query": query,
                            "answer": answer,
                            "timestamp": int(time.time())
                        }
                    )]
                )

            else:  # Local FAISS
                import faiss
                # Normalize for cosine similarity
                emb_array = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(emb_array)
                self.backend.add(emb_array)

                # Store metadata
                if session_id not in self.metadata:
                    self.metadata[session_id] = []
                self.metadata[session_id].append({
                    "query": query,
                    "answer": answer,
                    "timestamp": int(time.time())
                })

            LOGGER.info(
                "[FAISS] Added Q&A pair session=%s query_len=%d answer_len=%d",
                session_id, len(query), len(answer)
            )
            return True

        except Exception as e:
            LOGGER.exception("[FAISS][ADD][ERROR] %s", e)
            return False

    def search_similar(
        self,
        session_id: str,
        query: str,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search for similar Q&A pairs using vector similarity.

        Args:
            session_id: Session identifier
            query: Current query to find similar past queries
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
            # Generate embedding for the query
            embedding = self._get_embedding(query)
            if not embedding:
                LOGGER.warning("[FAISS] Cannot search without embedding")
                return []

            # Search based on backend type
            if isinstance(self.backend, _MemoryVectorBackend):
                results = self.backend.search(session_id, embedding, top_k)

            elif hasattr(self.backend, 'query'):  # Pinecone
                response = self.backend.query(
                    vector=embedding,
                    top_k=top_k,
                    filter={"session_id": session_id},
                    include_metadata=True
                )

                results = []
                for match in response.matches:
                    results.append({
                        "query": match.metadata.get("query", ""),
                        "answer": match.metadata.get("answer", ""),
                        "similarity": float(match.score),
                        "timestamp": match.metadata.get("timestamp", 0)
                    })

            else:  # Local FAISS
                import faiss
                # Normalize query embedding
                emb_array = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(emb_array)

                # Search
                distances, indices = self.backend.search(emb_array, top_k)

                # Filter by session and construct results
                results = []
                session_metadata = self.metadata.get(session_id, [])

                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0 and idx < len(session_metadata):
                        item = session_metadata[idx]
                        results.append({
                            "query": item["query"],
                            "answer": item["answer"],
                            "similarity": float(dist),
                            "timestamp": item["timestamp"]
                        })

            LOGGER.info(
                "[FAISS] Search completed session=%s query_len=%d results=%d",
                session_id, len(query), len(results)
            )
            return results

        except Exception as e:
            LOGGER.exception("[FAISS][SEARCH][ERROR] %s", e)
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
