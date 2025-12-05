# reader.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Driver
from common.settings import Neo4jConfig
from common.neo4j_repo import Neo4jRepository
from common.logger import Logger

LOGGER = Logger.get_logger("manager.reader")

class Reader:
    """
    Neo4j access layer for BPMN Graph-RAG.
    """

    def __init__(self, neo4j_config: Neo4jConfig,):
        try:
            LOGGER.warning("[FETCH][INIT] Injected repository invalid; falling back to Neo4jRepository(neo4j_config).")
            self.repository = Neo4jRepository(neo4j_config)
            LOGGER.info("[FETCH][INIT] Repository created from neo4j_config.")

            # DI health log
            LOGGER.info(
                "[FETCH][INIT] Reader initialized."
            )
        except Exception:
            # If initialization fails, bubble up; creation cannot proceed.
            Logger.get_logger(self.__class__.__name__).exception("[FETCH][INIT] Initialization failed.")
            raise

    # ------------------------------------------------------------------
    # 1) Hybrid search (consine)
    # ------------------------------------------------------------------
    def search_candidates(self, user_query: str, qemb: Optional[List[float]], limit: int = 200) -> List[Dict[str, Any]]:
        """
        One-shot retrieval:
        - Cosine via GDS gds.similarity.cosine(n.embedding_text, $qemb) if embedding exists.

        Requirements:
        - Fulltext index created:
            CREATE FULLTEXT INDEX nodeTextIndex IF NOT EXISTS FOR (n:Activity|Event|Gateway)
            ON EACH [n.name, n.summary_text, n.full_text];
        - n.embedding_text must be a float[] for cosine.
        - GDS library available.
        """
        try:
            LOGGER.info("[02.RETR] hybrid query limit=%d emb=%s", limit, qemb is not None)

            cypher = """
            // 1) Gather model keys under the bp node (optionally narrowed by $bp_id).
            MATCH (category)-[:CONTAINS_MODEL]->(m:BPMNModel)
            WITH collect(DISTINCT m.modelKey) AS model_keys

            // 2) Filter candidate nodes at MATCH-time using collected model_keys.
            MATCH (n:Activity|Event|Gateway)
            WHERE n.modelKey IN model_keys

            // 3) Cosine similarity when query embedding and node vector exist.
            WITH
            n,
            CASE
                WHEN $qemb IS NOT NULL AND n.context_vector IS NOT NULL
                     AND size(n.context_vector) > 0 AND size($qemb) > 0
                     AND size(n.context_vector) = size($qemb)
                THEN gds.similarity.cosine(n.context_vector, $qemb)
                ELSE 0.0
            END AS cos_sim

            // 4) Resolve owning process/lane/participant/model (participants may not exist).
            OPTIONAL MATCH (pr1:Process)-[:HAS_LANE]->(l:Lane)-[:OWNS_NODE]->(n)
            OPTIONAL MATCH (pr2:Process)-[:OWNS_NODE]->(n)
            WITH n, cos_sim, coalesce(pr1, pr2) AS pr, l

            OPTIONAL MATCH (part:Participant)-[:EXECUTES]->(pr)
            OPTIONAL MATCH (m:BPMNModel) WHERE m.modelKey = n.modelKey

            RETURN
                n.id                   AS node_id,
                head(labels(n))        AS node_label,
                n.name                 AS node_name,
                n.full_context         AS node_context,
                cos_sim,

                pr.id                  AS process_id,
                pr.name                AS process_name,
                pr.raw_context         AS process_context,

                l.id                   AS lane_id,
                l.name                 AS lane_name,
                l.summary_context      AS lane_context,

                part.id                AS part_id,
                part.name              AS part_name,
                part.summary_context   AS part_context,

                m.id                   AS model_id,
                m.name                 AS model_name,
                m.summary_context      AS model_context,
                m.modelKey             AS model_key
            ORDER BY cos_sim DESC
            LIMIT $limit;
            """
            rows = self.repository.execute_single_query(cypher, {"q": user_query, "qemb": qemb, "limit": int(limit)})
            return rows
        except Exception as e:
            LOGGER.exception("[02.RETR][HYBRID][ERROR] %s", e)
            return []

    # ------------------------------------------------------------------
    # 2) Model-level info/context & topology fetchers
    # ------------------------------------------------------------------
    def fetch_model_context(self, model_key: str) -> Dict[str, Any]:
        """
        Build a hierarchical JSON for an uploaded BPMN model:
        Model -> Participants -> Processes -> Lanes -> FlowNodes.

        Rules:
        - For Process level: return a "full_context" that mirrors reader_bpmn2neo.fetch_process_context
        (lanes, nodes, sequence/message flows, data I/O, annotations, groups, lane handoffs, paths).
        - For non-Process nodes (Model/Participant/Lane/FlowNode): return only id, name, and properties.
        - FlowNodes include both lane-owned nodes and process-owned nodes (the latter in 'flownodes' at process level).

        Returns:
        {
        "model": {"id","name","modelKey","properties"},
        "participants": [
            {
            "id","name","properties",
            "processes": [
                {
                "id",
                "name",
                "modelKey",
                "lanes": [
                    {
                    "id","name","properties",
                    "flownodes": [{"id","name"}]
                    }, ...
                ],
                "nodes_all": [
                    {
                    "id","name","properties",
                    "full_context",
                ],
                "message_flows": [],
                "data_reads": [],
                "data_writes": [],
                "annotations": [],
                "groups": [],
                "lane_handoffs": [],
                "paths_all": []
            ]
            }, ...
        ]
        }
        """
        logger = LOGGER
        logger.info("[UPLOAD_CTX] start model_key=%s", model_key)
        out: Dict[str, Any] = {}

        # ---------------------------
        # Defensive validation
        # ---------------------------
        try:
            if not model_key or not isinstance(model_key, str):
                logger.error("[UPLOAD_CTX][ERROR] invalid model_key")
                return {}
        except Exception:
            logger.exception("[UPLOAD_CTX][ERROR] validation failed")
            return {}

        # ---------------------------
        # 0) Fetch model meta
        # ---------------------------
        try:
            q_model = """
            MATCH (m:BPMNModel {modelKey:$mk})
            RETURN m.id AS id,
                coalesce(m.name,'BPMNModel '+toString(m.id)) AS name,
                m.modelKey AS modelKey
            """
            rows_model = self.repository.execute_single_query(q_model, {"mk": model_key}) or []
            if not rows_model:
                logger.warning("[UPLOAD_CTX] no model for key=%s", model_key)
                return {}
            mrow = rows_model[0]
            out["model"] = {
                "id": mrow.get("id"),
                "name": mrow.get("name"),
                "modelKey": mrow.get("modelKey"),
            }
            logger.info("[UPLOAD_CTX] model fetched id=%s", out["model"]["id"])
        except Exception:
            logger.exception("[UPLOAD_CTX][ERROR] model query failed")
            return {}

        # ---------------------------
        # 1) Fetch participants (id, name, properties) and their processes (id, name, properties)
        # ---------------------------
        try:
            q_part = """
            MATCH (m:BPMNModel {modelKey:$mk})-[:HAS_PARTICIPANT]->(pt:Participant)
            OPTIONAL MATCH (pt)-[:EXECUTES]->(pr:Process)
            WITH pt, collect(pr) AS prs
            RETURN pt.id AS pid,
                coalesce(pt.name,'Participant '+toString(pt.id)) AS pname,
                [p IN prs WHERE p IS NOT NULL |
                    { id: p.id, name: p.name}
                ] AS processes
            """
            rows_pt = self.repository.execute_single_query(q_part, {"mk": model_key}) or []
            participants: List[Dict[str, Any]] = []
            logger.info("[UPLOAD_CTX] participants count=%d", len(rows_pt))
        except Exception:
            logger.exception("[UPLOAD_CTX][ERROR] participant query failed")
            return out

        # ---------------------------
        # helper: collect all lanes (id, name, properties) for a process
        # ---------------------------
        def _lanes_for_process(model_key:str,pid: int) -> List[Dict[str, Any]]:
            try:
                q = """
                MATCH (pr:Process {id:$pid, modelKey:$modelKey})-[:HAS_LANE]->(l:Lane)
                RETURN l.id AS id, coalesce(l.name,'Lane '+toString(l.id)) AS name
                """
                rows = self.repository.execute_single_query(q, {"pid": pid, "modelKey":model_key}) or []
                return [{"id": r.get("id"), "name": r.get("name"), "properties": dict(r.get("props") or {})} for r in rows]
            except Exception:
                logger.exception("[UPLOAD_CTX][ERROR] lanes query failed pid=%s", pid)
                return []

        # ---------------------------
        # helper: flow-nodes under lane (Activity|Event|Gateway) => id, name, properties
        # ---------------------------
        def _flownodes_for_lane(model_key:str,lid: int) -> List[Dict[str, Any]]:
            try:
                q = """
                MATCH (l:Lane {id:$lid, modelKey:$modelKey})-[:OWNS_NODE]->(n:Activity|Event|Gateway)
                RETURN n.id AS id,
                    coalesce(n.name, head(labels(n))+' '+toString(n.id)) AS name,
                    properties(n) AS props
                """
                rows = self.repository.execute_single_query(q, {"lid": lid, "modelKey":model_key}) or []
                return [{"id": r.get("id"), "name": r.get("name"), "properties": dict(r.get("props") or {})} for r in rows]
            except Exception:
                logger.exception("[UPLOAD_CTX][ERROR] lane->nodes query failed lid=%s", lid)
                return []

        # ---------------------------
        # helper: process-owned (non-lane) flownodes => id, name, properties
        # ---------------------------
        def _process_owned_flownodes(model_key:str, pid: int) -> List[Dict[str, Any]]:
            try:
                q = """
                MATCH (pr:Process {id:$pid, modelKey:$modelKey})-[:OWNS_NODE]->(n:Activity|Event|Gateway)
                WHERE NOT ( (:Lane)-[:OWNS_NODE]->(n) )
                RETURN n.id AS id,
                       coalesce(n.name, head(labels(n))+' '+toString(n.id)) AS name
                """
                rows = self.repository.execute_single_query(q, {"pid": pid, "modelKey":model_key}) or []
                return [{"id": r.get("id"), "name": r.get("name"), "properties": dict(r.get("props") or {})} for r in rows]
            except Exception:
                logger.exception("[UPLOAD_CTX][ERROR] process->nodes(no-lane) query failed pid=%s", pid)
                return []

        # ---------------------------
        # helper: FULL process context (mirror of reader_bpmn2neo.fetch_process_context)
        #         - lanes, nodes(all/core), sequence/message flows, data I/O, annotations, groups, lane handoffs, paths
        # ---------------------------
        def _full_process_context(model_key:str, pid: int) -> Dict[str, Any]:
            ctx: Dict[str, Any] = {}
            try:
                # 1) process meta
                try:
                    q_meta = """
                    MATCH (pr:Process) WHERE pr.id=$pid and pr.modelKey=$modelKey
                    RETURN pr.id AS id, coalesce(pr.name,'Process '+toString(pr.id)) AS name, pr.modelKey AS modelKey
                    """
                    meta_rows = self.repository.execute_single_query(q_meta, {"pid": pid, "modelKey": model_key}) or []
                    meta = dict(meta_rows[0]) if meta_rows else {}
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] meta failed pid=%s", pid)
                    meta = {}
                    
                
                # 2) lanes (id, name, properties) and lane->node ids
                try:
                    q_lanes_ids = """
                    MATCH (pr:Process {id:$pid, modelKey:$modelKey})-[:HAS_LANE]->(l:Lane)
                    OPTIONAL MATCH (l)-[:OWNS_NODE]->(n:Activity|Event|Gateway)
                    RETURN l.id AS lid, coalesce(l.name,'Lane '+toString(l.id)) AS lname,
                        collect(n.id) AS nodeIds
                    """
                    rows_l = self.repository.execute_single_query(q_lanes_ids, {"pid": pid, "modelKey": model_key}) or []
                    lanes = [{
                        "id": r.get("lid"),
                        "name": r.get("lname"),
                        "node_ids": [nid for nid in (r.get("nodeIds") or []) if nid is not None],
                    } for r in rows_l]
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] lanes failed pid=%s", pid)
                    lanes = []
                

                # 3) all node ids (lane-owned + process-owned)
                try:
                    q_nodes_all = """
                    MATCH (pr:Process {id:$pid, modelKey:$modelKey})
                    OPTIONAL MATCH (pr)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(n1)
                    WITH pr, collect(n1) AS ns1
                    OPTIONAL MATCH (pr)-[:OWNS_NODE]->(n2)
                    WITH pr, ns1, collect(n2) AS ns2
                    WITH [x IN (ns1 + ns2) WHERE x IS NOT NULL] AS Ns
                    UNWIND Ns AS n
                    RETURN collect(DISTINCT n.id) AS nodeIds
                    """
                    rows_na = self.repository.execute_single_query(q_nodes_all, {"pid": pid, "modelKey": model_key}) or []
                    node_ids_all = rows_na[0].get("nodeIds") if rows_na else []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] nodes(all) failed pid=%s", pid)
                    node_ids_all = []

                # 4) core info for all nodes (union over Activity/Event/Gateway)
                try:
                    if node_ids_all:
                        q_core = """
                        // Activity
                        MATCH (a:Activity) WHERE a.id IN $ids and a.modelKey = $modelKey
                        OPTIONAL MATCH (l:Lane)-[:OWNS_NODE]->(a)
                        OPTIONAL MATCH (pr:Process)-[:OWNS_NODE]->(a)
                        RETURN a.id AS id, 'Activity' AS kind,
                            coalesce(a.name,'Activity '+toString(a.id)) AS name,
                            a.activityType AS activityType,
                            null AS position, null AS detailType,
                            null AS gatewayDirection, null AS gatewayDefault,
                            l.id AS ownerLaneId, coalesce(pr.id, null) AS ownerProcessId
                        UNION ALL
                        // Event
                        MATCH (e:Event) WHERE e.id IN $ids  and e.modelKey = $modelKey
                        OPTIONAL MATCH (l:Lane)-[:OWNS_NODE]->(e)
                        OPTIONAL MATCH (pr:Process)-[:OWNS_NODE]->(e)
                        RETURN e.id AS id, 'Event' AS kind,
                            coalesce(e.name,'Event '+toString(e.id)) AS name,
                            null AS activityType,
                            coalesce(e.position,'') AS position,
                            coalesce(e.detailType,'') AS detailType,
                            null AS gatewayDirection, null AS gatewayDefault,
                            l.id AS ownerLaneId, coalesce(pr.id, null) AS ownerProcessId
                        UNION ALL
                        // Gateway
                        MATCH (g:Gateway) WHERE g.id IN $ids and g.modelKey = $modelKey
                        OPTIONAL MATCH (l:Lane)-[:OWNS_NODE]->(g)
                        OPTIONAL MATCH (pr:Process)-[:OWNS_NODE]->(g)
                        RETURN g.id AS id, 'Gateway' AS kind,
                            coalesce(g.name,'Gateway '+toString(g.id)) AS name,
                            null AS activityType,
                            null AS position, null AS detailType,
                            coalesce(g.gatewayDirection,'') AS gatewayDirection,
                            coalesce(g.default,'') AS gatewayDefault,
                            l.id AS ownerLaneId, coalesce(pr.id, null) AS ownerProcessId
                        """
                        rows_core = self.repository.execute_single_query(q_core, {"ids": node_ids_all, "modelKey":model_key}) or []
                        nodes_all = [dict(r) for r in rows_core]
                    else:
                        nodes_all = []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] core nodes failed pid=%s", pid)
                    nodes_all = []

                # 5) lane handoffs (cross-lane sequence flows)
                try:
                    lane_handoffs = []
                    if node_ids_all:
                        q_handoffs = """
                        UNWIND $ids AS nid
                        WITH collect(nid) AS ids
                        MATCH (pr:Process)-[:HAS_LANE]->(ls:Lane)-[:OWNS_NODE]->(s:Activity|Event|Gateway)
                        MATCH (pr)-[:HAS_LANE]->(lt:Lane)-[:OWNS_NODE]->(t:Activity|Event|Gateway)
                        MATCH (s)-[r:SEQUENCE_FLOW]->(t)
                        WHERE s.id IN ids OR t.id IN ids
                        AND (s.modelKey = $modelKey and t.modelKey = $modelKey)
                        AND ls.id <> lt.id
                        RETURN
                            DISTINCT
                            r.id AS seqId,
                            s.id AS srcNode, coalesce(s.name, s.id) AS srcNodeName,
                            ls.id AS srcLane, coalesce(ls.name, 'Lane '+toString(ls.id)) AS srcLaneName,
                            t.id AS tgtNode, coalesce(t.name, t.id) AS tgtNodeName,
                            lt.id AS tgtLane, coalesce(lt.name, 'Lane '+toString(lt.id)) AS tgtLaneName,
                            pr.id AS srcProcessId, coalesce(pr.name, toString(pr.id)) AS srcProcessName,
                            pr.id AS tgtProcessId, coalesce(pr.name, toString(pr.id)) AS tgtProcessName,
                            coalesce(r.condition, r.properties.condition, '') AS condition,
                            coalesce(r.isDefault, r.properties.isDefault, false) AS isDefault
                        """
                        rows_h = self.repository.execute_single_query(q_handoffs, {"ids": node_ids_all, "modelKey":model_key}) or []
                        lane_handoffs = [dict(r) for r in rows_h]
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] lane handoffs failed pid=%s", pid)
                    lane_handoffs = []

                # 6) edges/attachments around these nodes
                # 6-1) sequence flows
                try:
                    if node_ids_all:
                        q_seq = """
                        MATCH (s)-[sf:SEQUENCE_FLOW]->(t)
                        WHERE s.id IN $ids OR t.id IN $ids
                        AND (s.modelKey = $modelKey and t.modelKey = $modelKey)
                        RETURN DISTINCT sf.id AS id, s.id AS src, t.id AS tgt,
                            coalesce(sf.isDefault,false) AS isDefault,
                            coalesce(sf.condition,'') AS condition,
                            coalesce(sf.flowName,'') AS flowName
                        """
                        rows_sf = self.repository.execute_single_query(q_seq, {"ids": node_ids_all, "modelKey": model_key}) or []
                        seq_flows = [dict(r) for r in rows_sf]
                    else:
                        seq_flows = []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] sequence flows failed pid=%s", pid)
                    seq_flows = []

                # 6-2) message flows
                try:
                    if node_ids_all:
                        q_mf = """
                        UNWIND $ids AS nid
                        WITH collect(nid) AS ids
                        MATCH (s)-[mf:MESSAGE_FLOW]->(t)
                        WHERE s.id IN ids OR t.id IN ids
                        AND (s.modelKey = $modelKey and t.modelKey = $modelKey)
                        OPTIONAL MATCH (pt_s:Participant)-[:EXECUTES]->(pr_s:Process)
                        WHERE (pr_s)-[:OWNS_NODE]->(s) OR (pr_s)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(s)
                        OPTIONAL MATCH (pt_t:Participant)-[:EXECUTES]->(pr_t:Process)
                        WHERE (pr_t)-[:OWNS_NODE]->(t) OR (pr_t)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(t)
                        RETURN
                            mf.id AS id, s.id AS src, t.id AS tgt,
                            coalesce(mf.flowName,'') AS flowName,
                            coalesce(mf.messageRef,'') AS messageRef,
                            coalesce(pt_s.id, elementId(pt_s)) AS srcParticipantId,
                            coalesce(pt_s.name, 'Unknown') AS srcParticipantName,
                            coalesce(pt_t.id, elementId(pt_t)) AS tgtParticipantId,
                            coalesce(pt_t.name, 'Unknown') AS tgtParticipantName
                        """
                        rows_mf = self.repository.execute_single_query(q_mf, {"ids": node_ids_all,"modelKey":model_key}) or []
                        msg_flows = [dict(r) for r in rows_mf]
                    else:
                        msg_flows = []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] message flows failed pid=%s", pid)
                    msg_flows = []

                # 6-3) data I/O (reads, writes)
                try:
                    if node_ids_all:
                        q_reads = """
                        MATCH (n:Activity)-[:READS_FROM]->(dr:DataReference)-[:REFERS_TO]->(d:Data)
                        WHERE n.id IN $ids AND n.modelKey = $modelKey
                        RETURN n.id AS node, dr.id AS dataRefId, d.id AS dataId,
                            coalesce(n.name,'') AS nodeName,
                            coalesce(dr.name,'DataRef '+toString(dr.id)) AS dataRefName,
                            coalesce(d.name,'Data '+toString(d.id)) AS dataName,
                            coalesce(dr.dataType,'ObjectReference') AS dataRefKind,
                            coalesce(dr.dataState,'ObjectReference') AS dataRefState,
                            coalesce(d.dataType,'Object') AS dataType,
                            coalesce(d.itemSubjectRef,'') AS itemSubjectRef,
                            coalesce(d.isCollection,false) AS isCollection,
                            coalesce(d.capacity, null) AS capacity
                        """
                        q_writes = """
                        MATCH (n:Activity)-[:WRITES_TO]->(dr:DataReference)-[:REFERS_TO]->(d:Data)
                        WHERE n.id IN $ids AND n.modelKey = $modelKey
                        RETURN n.id AS node, dr.id AS dataRefId, d.id AS dataId,
                            coalesce(n.name,'') AS nodeName,
                            coalesce(dr.name,'DataRef '+toString(dr.id)) AS dataRefName,
                            coalesce(d.name,'Data '+toString(d.id)) AS dataName,
                            coalesce(dr.dataType,'ObjectReference') AS dataRefKind,
                            coalesce(dr.dataState,'ObjectReference') AS dataRefState,
                            coalesce(d.dataType,'Object') AS dataType,
                            coalesce(d.itemSubjectRef,'') AS itemSubjectRef,
                            coalesce(d.isCollection,false) AS isCollection,
                            coalesce(d.capacity, null) AS capacity
                        """
                        reads_rows = self.repository.execute_single_query(q_reads, {"ids": node_ids_all,"modelKey": model_key}) or []
                        writes_rows = self.repository.execute_single_query(q_writes, {"ids": node_ids_all,"modelKey": model_key}) or []
                        data_reads = [dict(r) for r in reads_rows]
                        data_writes = [dict(r) for r in writes_rows]
                    else:
                        data_reads, data_writes = [], []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] data I/O failed pid=%s", pid)
                    data_reads, data_writes = [], []

                # 6-4) annotations
                try:
                    if node_ids_all:
                        q_ann = """
                        MATCH (ta:TextAnnotation)-[:ANNOTATES]->(x)
                        WHERE x.id IN $ids AND x.modelKey = $modelKey
                        RETURN ta.id AS id, coalesce(ta.text,'') AS text, x.id AS targetId
                        """
                        rows_ann = self.repository.execute_single_query(q_ann, {"ids": node_ids_all,"modelKey": model_key}) or []
                        annotations = [dict(r) for r in rows_ann]
                    else:
                        annotations = []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] annotations failed pid=%s", pid)
                    annotations = []

                # 6-5) groups
                try:
                    if node_ids_all:
                        q_grp = """
                        MATCH (g:Group)-[:GROUPS]->(m)
                        WHERE m.id IN $ids AND m.modelKey = $modelKey
                        RETURN g.id AS id, coalesce(g.name,'Group '+toString(g.id)) AS name, m.id AS memberId
                        """
                        rows_grp = self.repository.execute_single_query(q_grp, {"ids": node_ids_all, "modelKey": model_key}) or []
                        groups = [dict(r) for r in rows_grp]
                    else:
                        groups = []
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] groups failed pid=%s", pid)
                    groups = []

                # 7) Paths (BFS shortest paths from Start to End inside the process)
                try:
                    q_paths = """
                    WITH $pid AS pid
                    MATCH (p:Process {id: pid, modelKey: $modelKey})
                    OPTIONAL MATCH (p)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(n1)
                    WITH p, collect(n1) AS ns1
                    OPTIONAL MATCH (p)-[:OWNS_NODE]->(n2)
                    WITH p, ns1, collect(n2) AS ns2
                    WITH p, [x IN (ns1 + ns2) WHERE x IS NOT NULL] AS N_all
                    WITH p, N_all,
                        [n IN N_all
                        WHERE (n:Event AND toLower(coalesce(n.position,''))='start')
                            OR size([x IN N_all WHERE (x)-[:SEQUENCE_FLOW]->(n)]) = 0] AS starts,
                        [n IN N_all
                        WHERE (n:Event AND toLower(coalesce(n.position,''))='end')
                            OR size([x IN N_all WHERE (n)-[:SEQUENCE_FLOW]->(x)]) = 0] AS ends
                    UNWIND starts AS s
                    CALL apoc.path.expandConfig(s, {
                        relationshipFilter: 'SEQUENCE_FLOW>|HAS_BOUNDARY_EVENT>',
                        bfs: true,
                        whitelistNodes: N_all,
                        terminatorNodes: ends,
                        uniqueness: 'NODE_GLOBAL',
                        filterStartNode: true,
                        maxLevel: -1
                    }) YIELD path
                    WITH ends, path
                    WHERE last(nodes(path)) IN ends
                    WITH path
                    ORDER BY length(path) ASC
                    LIMIT $maxPaths
                    RETURN DISTINCT [x IN nodes(path) | {id: x.id, name: coalesce(x.name, x.id)}] AS nodePath
                    """
                    rows_paths = self.repository.execute_single_query(q_paths, {"pid": pid, "maxPaths": 100, "modelKey":model_key}) or []
                    main_paths = [r.get("nodePath") for r in rows_paths if r.get("nodePath")]
                except Exception:
                    logger.exception("[UPLOAD_CTX][PROC] paths failed pid=%s", pid)
                    main_paths = []

                ctx = {
                    "id": meta['id'],
                    "name":meta['name'],
                    "lanes": lanes,
                    "nodes_all": nodes_all,
                    "sequence_flows": seq_flows,
                    "message_flows": msg_flows,
                    "data_reads": data_reads,
                    "data_writes": data_writes,
                    "annotations": annotations,
                    "groups": groups,
                    "lane_handoffs": lane_handoffs,
                    "paths_all": main_paths,
                }
                return ctx
            except Exception:
                logger.exception("[UPLOAD_CTX][PROC] FAILED pid=%s", pid)
                return {}

        # ---------------------------
        # 2) Assemble hierarchy
        # ---------------------------
        try:
            for r in rows_pt:
                pt_obj = {
                    "id": r.get("pid"),
                    "name": r.get("pname"),
                    "processes": []
                }

                # each process under participant
                proc_list = r.get("processes") or []
                logger.info("[UPLOAD_CTX] participant id=%s processes=%d", pt_obj["id"], len(proc_list))
                for p in proc_list:
                    pid = p.get("id")
                    #pname = p.get("name")

                    # lanes -> flownodes
                    #lanes = _lanes_for_process(model_key,pid)
                    #for ln in lanes:
                    #    ln["flownodes"] = _flownodes_for_lane(model_key,ln["id"])

                    # process-owned flownodes (no lane)
                    # flownodes_no_lane = _process_owned_flownodes(model_key,pid)

                    # full context for process
                    full_ctx = _full_process_context(model_key,pid)

                    pt_obj["processes"].append(full_ctx)

                participants.append(pt_obj)

            out["participants"] = participants
            logger.info("[UPLOAD_CTX] done participants=%d", len(participants))
            return out
        except Exception:
            logger.exception("[UPLOAD_CTX][ERROR] assemble hierarchy failed")
            return out


    # ------------------------------------------------------------------
    # ETC) Graph Views for Streamlit-Agraph
    # ------------------------------------------------------------------
    def get_overall_process_graph(self, model_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        View 1) Overall Process Graph (Activity|Event|Gateway + SEQUENCE_FLOW)
        Rules aligned with parser/loader:
          - Model scope: (m:BPMNModel {modelKey:$mk}) → Participants → Processes
          - Nodes: Activity|Event|Gateway owned by either Lane or Process
          - Edges: SEQUENCE_FLOW within same model
        Returns dict for agraph: {"nodes":[...], "edges":[...]} with node.id from property n.id.
        """
        try:
            cypher = """
            // Collect flownodes by modelKey (no assumption about Participant/Process/Lane presence)
            CALL {
            WITH $mk AS mk
            // A) Include all flownodes (so isolated/end nodes also appear)
            MATCH (n:Activity|Event|Gateway {modelKey:mk})
            RETURN DISTINCT
                n.id AS src_id, n.name AS src_name, labels(n) AS src_labels,
                NULL AS tgt_id, NULL AS tgt_name, NULL AS tgt_labels,
                NULL AS rel_type

            UNION
            // B) SEQUENCE_FLOW edges within the same model
            MATCH (n:Activity|Event|Gateway {modelKey:$mk})-[r:SEQUENCE_FLOW]->(t:Activity|Event|Gateway {modelKey:$mk})
            RETURN
                n.id AS src_id, n.name AS src_name, labels(n) AS src_labels,
                t.id AS tgt_id, t.name AS tgt_name, labels(t) AS tgt_labels,
                type(r) AS rel_type

            UNION
            // C) HAS_BOUNDARY_EVENT edges (boundary event attached to a host activity)
            MATCH (n:Activity {modelKey:$mk})-[r:HAS_BOUNDARY_EVENT]->(t:Event {modelKey:$mk})
            RETURN
                n.id AS src_id, n.name AS src_name, labels(n) AS src_labels,
                t.id AS tgt_id, t.name AS tgt_name, labels(t) AS tgt_labels,
                type(r) AS rel_type
            }
            RETURN DISTINCT src_id, src_name, src_labels, tgt_id, tgt_name, tgt_labels, rel_type
            """
            rows = self.repository.execute_single_query(cypher, {"mk": model_key})

            node_map: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            def upsert(nid: Any, name: Any, labs: List[str]) -> None:
                if nid is None:
                    return
                k = str(nid)
                if k not in node_map:
                    lbl = name or k
                    group = next((l for l in (labs or []) if l in ("Activity", "Event", "Gateway")), "Node")
                    node_map[k] = {"id": k, "label": lbl, "title": ", ".join(labs or []), "group": group}

            for r in rows:
                upsert(r.get("src_id"), r.get("src_name"), r.get("src_labels") or [])
                if r.get("tgt_id"):
                    upsert(r.get("tgt_id"), r.get("tgt_name"), r.get("tgt_labels") or [])
                    edges.append({
                        "source": str(r["src_id"]),
                        "target": str(r["tgt_id"]),
                        "label": r.get("rel_type") or "SEQUENCE_FLOW",
                    })

            return {"nodes": list(node_map.values()), "edges": edges}
        except Exception as e:
            LOGGER.exception("[READER][OVERALL][ERROR] %s", e)
            return {"nodes": [], "edges": []}

    def get_message_exchange_graph(self, model_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        View 2) Participant Message Exchange Graph
        Nodes:
          - Participants
          - Involved Activities/Events (endpoints of MESSAGE_FLOW)
        Edges:
          - MESSAGE_FLOW between node→node
          - Visual helper edges 'OWNS' from Participant→node
        Participant resolution per parser/loader:
          Participant -[:EXECUTES]-> Process
          Process -[:HAS_LANE]-> Lane -[:OWNS_NODE]-> node  (or Process -[:OWNS_NODE]-> node)
        """
        try:
            cypher = """
            // MESSAGE_FLOW endpoints in model
            MATCH (s:Activity|Event {modelKey:$mk})-[mf:MESSAGE_FLOW]->(t:Activity|Event {modelKey:$mk})
            // Resolve s's participant via Process/Lane
            OPTIONAL MATCH (pr1:Process)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(s)
            OPTIONAL MATCH (pr1d:Process)-[:OWNS_NODE]->(s)
            WITH s, t, mf, coalesce(pr1, pr1d) AS pr1x
            OPTIONAL MATCH (p1:Participant)-[:EXECUTES]->(pr1x)
            // Resolve t's participant via Process/Lane
            OPTIONAL MATCH (pr2:Process)-[:HAS_LANE]->(:Lane)-[:OWNS_NODE]->(t)
            OPTIONAL MATCH (pr2d:Process)-[:OWNS_NODE]->(t)
            WITH s, t, mf, p1, coalesce(pr2, pr2d) AS pr2x
            OPTIONAL MATCH (p2:Participant)-[:EXECUTES]->(pr2x)
            RETURN DISTINCT
              s.id AS s_id, s.name AS s_name, labels(s) AS s_labels,
              t.id AS t_id, t.name AS t_name, labels(t) AS t_labels,
              CASE WHEN p1 IS NULL THEN NULL ELSE p1.id END AS p1_id,
              CASE WHEN p1 IS NULL THEN NULL ELSE p1.name END AS p1_name,
              CASE WHEN p2 IS NULL THEN NULL ELSE p2.id END AS p2_id,
              CASE WHEN p2 IS NULL THEN NULL ELSE p2.name END AS p2_name,
              type(mf) AS rel_type
            """
            rows = self.repository.execute_single_query(cypher, {"mk": model_key})

            node_map: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            def add_node(nid: Any, name: Any, labs: List[str], group_hint: Optional[str] = None) -> None:
                if nid is None:
                    return
                k = str(nid)
                if k not in node_map:
                    node_map[k] = {
                        "id": k,
                        "label": (name or k),
                        "title": ", ".join(labs or []),
                        "group": (group_hint or next((l for l in (labs or []) if l in ("Activity", "Event", "Participant")), "Node")),
                    }

            for r in rows:
                add_node(r.get("s_id"), r.get("s_name"), r.get("s_labels") or [])
                add_node(r.get("t_id"), r.get("t_name"), r.get("t_labels") or [])
                if r.get("p1_id"):
                    add_node(r.get("p1_id"), r.get("p1_name"), ["Participant"], "Participant")
                if r.get("p2_id"):
                    add_node(r.get("p2_id"), r.get("p2_name"), ["Participant"], "Participant")

                # MESSAGE_FLOW edge
                if r.get("s_id") and r.get("t_id"):
                    edges.append({"source": str(r["s_id"]), "target": str(r["t_id"]), "label": r.get("rel_type") or "MESSAGE_FLOW"})
                # Visual helper: Participant -> Node
                if r.get("p1_id") and r.get("s_id"):
                    edges.append({"source": str(r["p1_id"]), "target": str(r["s_id"]), "label": "OWNS"})
                if r.get("p2_id") and r.get("t_id"):
                    edges.append({"source": str(r["p2_id"]), "target": str(r["t_id"]), "label": "OWNS"})

            return {"nodes": list(node_map.values()), "edges": edges}
        except Exception as e:
            LOGGER.exception("[READER][MESSAGE][ERROR] %s", e)
            return {"nodes": [], "edges": []}

    def get_subprocess_graph(self, model_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        View 3) Subprocess Graph
        Parser rule:
          - SubProcess/AdHocSubProcess/Transaction are Activity nodes with activityType property.
          - Subprocess internal FlowNodes are linked via (sp:Activity)-[:CONTAINS]->(n:Activity|Event|Gateway)
        We show:
          - CONTAINS edges sp → inner
          - Internal SEQUENCE_FLOW among inner nodes (only when both are contained by the same sp)
        """
        try:
            cypher = """
            // SubProcess hosts in model
            MATCH (sp:Activity {modelKey:$mk})
            WHERE sp.activityType IN ['SubProcess','AdHocSubProcess','Transaction']
            // Contained inner nodes
            OPTIONAL MATCH (sp)-[:CONTAINS]->(n:Activity|Event|Gateway)
            WITH sp, n
            OPTIONAL MATCH (n)-[r:SEQUENCE_FLOW]->(t:Activity|Event|Gateway)
            WHERE (sp)-[:CONTAINS]->(t)
            RETURN DISTINCT
              sp.id AS sp_id, sp.name AS sp_name, labels(sp) AS sp_labels,
              n.id AS n_id, n.name AS n_name, labels(n) AS n_labels,
              t.id AS t_id, t.name AS t_name, labels(t) AS t_labels,
              CASE WHEN r IS NULL THEN NULL ELSE type(r) END AS rel_type
            """
            rows = self.repository.execute_single_query(cypher, {"mk": model_key})

            node_map: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            def add(nid: Any, name: Any, labs: List[str], group: str) -> None:
                if nid is None:
                    return
                k = str(nid)
                if k not in node_map:
                    node_map[k] = {"id": k, "label": (name or k), "title": ", ".join(labs or []), "group": group}

            for r in rows:
                add(r.get("sp_id"), r.get("sp_name"), r.get("sp_labels") or [], "SubProcess")
                if r.get("n_id"):
                    add(r.get("n_id"), r.get("n_name"), r.get("n_labels") or [], "InnerNode")
                    # containment edge
                    edges.append({"source": str(r["sp_id"]), "target": str(r["n_id"]), "label": "CONTAINS"})
                if r.get("t_id") and r.get("n_id"):
                    add(r.get("t_id"), r.get("t_name"), r.get("t_labels") or [], "InnerNode")
                    edges.append({"source": str(r["n_id"]), "target": str(r["t_id"]), "label": r.get("rel_type") or "SEQUENCE_FLOW"})

            return {"nodes": list(node_map.values()), "edges": edges}
        except Exception as e:
            LOGGER.exception("[READER][SUBPROC][ERROR] %s", e)
            return {"nodes": [], "edges": []}

    def get_data_io_graph(self, model_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        View 4) Data I/O Graph
        Parser rule:
          - Activity|Event --[READS_FROM|WRITES_TO]--> DataReference
          - DataReference --[REFERS_TO]--> Data   (Data nodes have property dataType: 'Object' or 'Store')
        We show:
          - READS_FROM / WRITES_TO edges to DataReference
          - REFERS_TO edge to physical Data (optional)
        """
        try:
            cypher = """
            MATCH (n:Activity|Event {modelKey:$mk})-[r:READS_FROM|WRITES_TO]->(dr:DataReference {modelKey:$mk})
            OPTIONAL MATCH (dr)-[rt:REFERS_TO]->(d:Data {modelKey:$mk})
            RETURN DISTINCT
              n.id AS n_id, n.name AS n_name, labels(n) AS n_labels,
              type(r) AS io_type,
              dr.id AS dr_id, dr.name AS dr_name, labels(dr) AS dr_labels,
              d.id AS d_id, d.name AS d_name, labels(d) AS d_labels
            """
            rows = self.repository.execute_single_query(cypher, {"mk": model_key})

            node_map: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            def add_node(nid: Any, name: Any, labs: List[str], group_hint: str) -> None:
                if nid is None:
                    return
                k = str(nid)
                if k not in node_map:
                    node_map[k] = {"id": k, "label": (name or k), "title": ", ".join(labs or []), "group": group_hint}

            for r in rows:
                add_node(r.get("n_id"), r.get("n_name"), r.get("n_labels") or [], "FlowNode")
                add_node(r.get("dr_id"), r.get("dr_name"), r.get("dr_labels") or [], "DataRef")
                edges.append({"source": str(r["n_id"]), "target": str(r["dr_id"]), "label": r.get("io_type")})
                if r.get("d_id"):
                    add_node(r.get("d_id"), r.get("d_name"), r.get("d_labels") or [], "Data")
                    edges.append({"source": str(r["dr_id"]), "target": str(r["d_id"]), "label": "REFERS_TO"})

            return {"nodes": list(node_map.values()), "edges": edges}
        except Exception as e:
            LOGGER.exception("[READER][DATAIO][ERROR] %s", e)
            return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # 5) Category & Model Tree & Hierarchy
    # ------------------------------------------------------------------
    def fetch_category_tree_only(self, container_id: str) -> List[Dict[str, Any]]:
        """
        카테고리 노드만 계층 구조로 조회 (업로더에서 사용)

        구조:
        Container → HAS_CATEGORY → Category (최상위)
                                   └→ HAS_SUBCATEGORY → Category (하위) → ...

        Returns:
            st_ant_tree 형식의 트리 데이터 (카테고리만)
        """
        try:
            LOGGER.info("[CATEGORY_TREE] Fetching category-only tree container_id=%s", container_id)

            cypher = """
            // 1) 루트 컨테이너 조회
            MATCH (c {id: $container_id})

            // 2) 컨테이너가 직접 가지고 있는 최상위 카테고리 노드
            WITH c
            MATCH (c)-[:HAS_CATEGORY]->(root_cat:Category)

            // 3) 재귀적으로 HAS_SUBCATEGORY 관계의 하위 카테고리 조회
            OPTIONAL MATCH path = (root_cat)-[:HAS_SUBCATEGORY*0..]->(child_cat:Category)

            WITH root_cat, child_cat, path
            WHERE child_cat IS NOT NULL

            RETURN DISTINCT
                child_cat.id AS id,
                child_cat.modelKey AS model_key,
                child_cat.name AS name,
                CASE
                    WHEN path IS NULL THEN [root_cat.modelKey]
                    ELSE [root_cat.modelKey] + [node IN nodes(path)[1..] | node.modelKey]
                END AS path_keys
            ORDER BY path_keys, child_cat.name
            """

            rows = self.repository.execute_single_query(cypher, {"container_id": container_id})

            LOGGER.info("[CATEGORY_TREE] Query returned %d category nodes", len(rows))

            # 트리 구조로 변환
            tree_data = self._build_category_tree_structure(rows)

            LOGGER.info("[CATEGORY_TREE] Built tree with %d root nodes", len(tree_data))
            return tree_data

        except Exception as e:
            LOGGER.exception("[CATEGORY_TREE][ERROR] %s", e)
            return []

    def fetch_category_tree_with_models(self, container_id: str) -> List[Dict[str, Any]]:
        """
        카테고리 + 하위 모델까지 포함한 계층 구조 조회 (패널에서 사용)

        구조:
        Container → Category → CONTAINS_MODEL → Model
                           └→ HAS_SUBCATEGORY → Category → CONTAINS_MODEL → Model

        Returns:
            st_ant_tree 형식의 트리 데이터 (카테고리 + 모델)
        """
        try:
            LOGGER.info("[CATEGORY_TREE_FULL] Fetching category + models tree container_id=%s", container_id)

            # 1) 카테고리 트리 먼저 가져오기
            category_tree = self.fetch_category_tree_only(container_id)

            # 2) 각 카테고리에 CONTAINS_MODEL 관계로 연결된 Model 추가
            self._attach_models_to_categories(category_tree)

            LOGGER.info("[CATEGORY_TREE_FULL] Complete tree built")
            return category_tree

        except Exception as e:
            LOGGER.exception("[CATEGORY_TREE_FULL][ERROR] %s", e)
            return []

    def fetch_models_under_category(self, category_key: str) -> List[Dict[str, Any]]:
        """
        특정 카테고리 하위의 모델 목록 조회 (NEXT_PROCESS 관계 포함)

        Args:
            category_key: 카테고리 modelKey

        Returns:
            모델 리스트 (NEXT_PROCESS 관계 정보 포함)
        """
        try:
            LOGGER.info("[CATEGORY_MODELS] Fetching models under category=%s", category_key)

            cypher = """
            MATCH (cat:Category {modelKey: $category_key})-[:CONTAINS_MODEL]->(m:BPMNModel)

            // NEXT_PROCESS 관계 조회
            OPTIONAL MATCH (m)-[:NEXT_PROCESS]->(next:BPMNModel)
            OPTIONAL MATCH (prev:BPMNModel)-[:NEXT_PROCESS]->(m)

            RETURN
                m.id AS id,
                m.modelKey AS model_key,
                m.name AS name,
                next.modelKey AS next_model_key,
                prev.modelKey AS prev_model_key
            ORDER BY m.name
            """

            rows = self.repository.execute_single_query(cypher, {"category_key": category_key})

            LOGGER.info("[CATEGORY_MODELS] Found %d models", len(rows))
            return rows

        except Exception as e:
            LOGGER.exception("[CATEGORY_MODELS][ERROR] %s", e)
            return []

    def get_models_with_next_process(self, model_keys: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        모델 리스트에 대한 NEXT_PROCESS 관계 그래프 조회

        Args:
            model_keys: 조회할 모델 키 리스트

        Returns:
            agraph 형식 {"nodes": [...], "edges": [...]}
        """
        try:
            LOGGER.info("[NEXT_PROCESS_GRAPH] Fetching graph for %d models", len(model_keys))

            cypher = """
            // 요청된 모델들과 그들의 NEXT_PROCESS 관계
            MATCH (m1:BPMNModel)-[:NEXT_PROCESS]->(m2:BPMNModel)
            WHERE m1.modelKey IN $keys OR m2.modelKey IN $keys

            RETURN
                m1.modelKey AS src_key,
                m1.name AS src_name,
                m2.modelKey AS tgt_key,
                m2.name AS tgt_name
            """

            rows = self.repository.execute_single_query(cypher, {"keys": model_keys})

            node_map: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            for r in rows:
                src_key = r.get("src_key")
                tgt_key = r.get("tgt_key")

                # 노드 추가
                if src_key and src_key not in node_map:
                    node_map[src_key] = {
                        "id": src_key,
                        "label": r.get("src_name") or src_key,
                        "title": r.get("src_name") or src_key,
                        "group": "Model"
                    }

                if tgt_key and tgt_key not in node_map:
                    node_map[tgt_key] = {
                        "id": tgt_key,
                        "label": r.get("tgt_name") or tgt_key,
                        "title": r.get("tgt_name") or tgt_key,
                        "group": "Model"
                    }

                # 엣지 추가
                if src_key and tgt_key:
                    edges.append({
                        "source": src_key,
                        "target": tgt_key,
                        "label": "NEXT"
                    })

            LOGGER.info("[NEXT_PROCESS_GRAPH] Built graph with %d nodes, %d edges",
                       len(node_map), len(edges))

            return {"nodes": list(node_map.values()), "edges": edges}

        except Exception as e:
            LOGGER.exception("[NEXT_PROCESS_GRAPH][ERROR] %s", e)
            return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Helper methods for category tree building
    # ------------------------------------------------------------------
    def _build_category_tree_structure(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        쿼리 결과를 st_ant_tree 형식으로 변환 (카테고리만)

        Args:
            rows: Neo4j 쿼리 결과

        Returns:
            st_ant_tree 형식 트리 데이터
        """
        try:
            node_map = {}
            root_nodes = []

            for row in rows:
                model_key = row.get("model_key")
                if not model_key:
                    continue

                path_keys = row.get("path_keys", [])

                node = {
                    "value": model_key,
                    "title": row.get("name") or model_key,
                    "children": [],
                    "is_category": True,
                    "_raw": row  # 원본 데이터 보관 (디버깅용)
                }

                node_map[model_key] = node

                # 루트 노드 판별 (path_keys 길이가 1)
                if len(path_keys) == 1:
                    root_nodes.append(node)
                elif len(path_keys) > 1:
                    # 부모 노드의 children에 추가
                    parent_key = path_keys[-2]
                    if parent_key in node_map:
                        # 중복 추가 방지
                        if node not in node_map[parent_key]["children"]:
                            node_map[parent_key]["children"].append(node)

            LOGGER.debug("[TREE_BUILD] Built %d root nodes from %d total nodes",
                        len(root_nodes), len(node_map))

            return root_nodes

        except Exception as e:
            LOGGER.exception("[TREE_BUILD][ERROR] %s", e)
            return []

    def _attach_models_to_categories(self, category_tree: List[Dict[str, Any]]) -> None:
        """
        카테고리 트리의 각 카테고리 노드에 CONTAINS_MODEL 관계의 Model 추가

        Args:
            category_tree: 카테고리 트리 (수정됨 - in-place)
        """
        try:
            for category_node in category_tree:
                category_key = category_node.get("value")

                if category_key:
                    # 해당 카테고리의 하위 모델 조회
                    models = self.fetch_models_under_category(category_key)

                    # 모델을 children에 추가
                    for model in models:
                        model_node = {
                            "value": model.get("model_key"),
                            "title": model.get("name") or model.get("model_key"),
                            "is_category": False,
                            "next_model_key": model.get("next_model_key"),
                            "prev_model_key": model.get("prev_model_key"),
                            "_raw": model
                        }
                        category_node["children"].append(model_node)

                # 하위 카테고리에 대해서도 재귀 처리
                if "children" in category_node and category_node["children"]:
                    child_categories = [c for c in category_node["children"] if c.get("is_category")]
                    if child_categories:
                        self._attach_models_to_categories(child_categories)

            LOGGER.debug("[ATTACH_MODELS] Completed attaching models to categories")

        except Exception as e:
            LOGGER.exception("[ATTACH_MODELS][ERROR] %s", e)
