"""Neo4j Knowledge Graph — entity & classification graph (MASTER_PROMPT §6.4).

Provides a graph-backed store for:
- **Document classification relationships**: Documents linked to their document_type
  and classification_label nodes, enabling graph-based similarity discovery
- **Entity graph**: Canonical entities with aliases, mentions linked to chunks/documents
- **Classification evolution**: Tracks classification lineage (which method, confidence)
  for self-learning feedback loops

Node types:
    (:Document {doc_id, filename})
    (:DocumentType {name})          — e.g., "10-K", "annual_report"
    (:ClassificationLabel {name})   — e.g., "sec_filing", "financial_report"
    (:Entity {entity_id, canonical_name, entity_type, aliases})

Relationship types:
    (Document)-[:CLASSIFIED_AS {confidence, method, timestamp}]->(DocumentType)
    (Document)-[:HAS_LABEL {confidence}]->(ClassificationLabel)
    (DocumentType)-[:BELONGS_TO]->(ClassificationLabel)
    (Entity)-[:MENTIONED_IN {mention_text, confidence, page_numbers}]->(Document)
    (Document)-[:SIMILAR_TO {similarity}]->(Document)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

# ── Neo4j Driver Singleton ──────────────────────────────────────────────

_driver = None


def _get_driver():
    """Lazily create the Neo4j driver singleton."""
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
            )
            logger.info("Neo4j driver connected to %s", settings.neo4j.uri)
        except Exception as exc:
            logger.warning("Neo4j connection failed: %s", exc)
            raise
    return _driver


def close_driver() -> None:
    """Close the Neo4j driver. Call on application shutdown."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextmanager
def get_session() -> Iterator:
    """Yield a Neo4j session from the driver pool."""
    driver = _get_driver()
    session = driver.session(database=settings.neo4j.database)
    try:
        yield session
    finally:
        session.close()


def reset_driver_for_testing() -> None:
    """Reset driver singleton for test isolation."""
    global _driver
    _driver = None


# ── Schema Initialization ───────────────────────────────────────────────

def init_schema() -> None:
    """Create indexes and constraints in Neo4j. Idempotent."""
    with get_session() as session:
        session.run(
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT doc_type_unique IF NOT EXISTS "
            "FOR (dt:DocumentType) REQUIRE dt.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT class_label_unique IF NOT EXISTS "
            "FOR (cl:ClassificationLabel) REQUIRE cl.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX doc_filename IF NOT EXISTS "
            "FOR (d:Document) ON (d.filename)"
        )
        logger.info("Neo4j schema initialized")


# ── Document Classification Graph ───────────────────────────────────────

def store_classification(
    doc_id: str,
    filename: str,
    document_type: str,
    classification_label: str,
    confidence: float,
    method: str,
    page_count: int = 0,
) -> None:
    """Store a document classification in the knowledge graph.

    Creates/merges:
    - Document node
    - DocumentType node
    - ClassificationLabel node
    - CLASSIFIED_AS relationship (Document → DocumentType)
    - HAS_LABEL relationship (Document → ClassificationLabel)
    - BELONGS_TO relationship (DocumentType → ClassificationLabel)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        session.run(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.filename = $filename, d.page_count = $page_count,
                d.last_classified = $timestamp
            MERGE (dt:DocumentType {name: $document_type})
            MERGE (cl:ClassificationLabel {name: $classification_label})
            MERGE (dt)-[:BELONGS_TO]->(cl)
            MERGE (d)-[r:CLASSIFIED_AS]->(dt)
            SET r.confidence = $confidence, r.method = $method,
                r.timestamp = $timestamp
            MERGE (d)-[rl:HAS_LABEL]->(cl)
            SET rl.confidence = $confidence
            """,
            doc_id=doc_id,
            filename=filename,
            document_type=document_type,
            classification_label=classification_label,
            confidence=confidence,
            method=method,
            page_count=page_count,
            timestamp=timestamp,
        )


def find_similar_documents(
    document_type: str,
    classification_label: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Find documents with the same classification.

    Returns documents sharing the same DocumentType or ClassificationLabel,
    useful for cross-document analysis and classification reinforcement.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (d:Document)-[r:CLASSIFIED_AS]->(dt:DocumentType {name: $document_type})
            RETURN d.doc_id AS doc_id, d.filename AS filename,
                   r.confidence AS confidence, r.method AS method
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            document_type=document_type,
            limit=limit,
        )
        return [dict(record) for record in result]


def find_documents_by_label(
    classification_label: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Find all documents with a given classification label."""
    with get_session() as session:
        result = session.run(
            """
            MATCH (d:Document)-[:HAS_LABEL]->(cl:ClassificationLabel {name: $label})
            OPTIONAL MATCH (d)-[r:CLASSIFIED_AS]->(dt:DocumentType)
            RETURN d.doc_id AS doc_id, d.filename AS filename,
                   dt.name AS document_type, r.confidence AS confidence
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            label=classification_label,
            limit=limit,
        )
        return [dict(record) for record in result]


def get_classification_graph_stats() -> Dict[str, Any]:
    """Return summary statistics of the classification knowledge graph."""
    with get_session() as session:
        result = session.run(
            """
            MATCH (d:Document) WITH count(d) AS doc_count
            MATCH (dt:DocumentType) WITH doc_count, count(dt) AS type_count
            MATCH (cl:ClassificationLabel) WITH doc_count, type_count, count(cl) AS label_count
            RETURN doc_count, type_count, label_count
            """
        )
        record = result.single()
        if record:
            return {
                "document_count": record["doc_count"],
                "document_type_count": record["type_count"],
                "classification_label_count": record["label_count"],
            }
        return {"document_count": 0, "document_type_count": 0, "classification_label_count": 0}


def get_type_distribution() -> List[Dict[str, Any]]:
    """Get distribution of document types in the graph."""
    with get_session() as session:
        result = session.run(
            """
            MATCH (d:Document)-[:CLASSIFIED_AS]->(dt:DocumentType)
            RETURN dt.name AS document_type, count(d) AS count,
                   avg(d.page_count) AS avg_pages
            ORDER BY count DESC
            """
        )
        return [dict(record) for record in result]


# ── Entity Knowledge Graph ──────────────────────────────────────────────

def store_entity(
    entity_id: str,
    canonical_name: str,
    entity_type: str,
    aliases: Optional[List[str]] = None,
) -> None:
    """Create or update an entity node in the knowledge graph."""
    with get_session() as session:
        session.run(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.canonical_name = $canonical_name,
                e.entity_type = $entity_type,
                e.aliases = $aliases
            """,
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases or [],
        )


def store_entity_mention(
    entity_id: str,
    doc_id: str,
    mention_text: str,
    confidence: float,
    page_numbers: Optional[List[int]] = None,
) -> None:
    """Link an entity mention to a document in the graph."""
    with get_session() as session:
        session.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (d:Document {doc_id: $doc_id})
            MERGE (e)-[r:MENTIONED_IN]->(d)
            SET r.mention_text = $mention_text,
                r.confidence = $confidence,
                r.page_numbers = $page_numbers
            """,
            entity_id=entity_id,
            doc_id=doc_id,
            mention_text=mention_text,
            confidence=confidence,
            page_numbers=page_numbers or [],
        )


def find_entity_documents(
    entity_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Find all documents where an entity is mentioned."""
    with get_session() as session:
        result = session.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})-[r:MENTIONED_IN]->(d:Document)
            RETURN d.doc_id AS doc_id, d.filename AS filename,
                   r.mention_text AS mention_text, r.confidence AS confidence,
                   r.page_numbers AS page_numbers
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            entity_id=entity_id,
            limit=limit,
        )
        return [dict(record) for record in result]


def find_entities_in_document(
    doc_id: str,
) -> List[Dict[str, Any]]:
    """Find all entities mentioned in a document."""
    with get_session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[r:MENTIONED_IN]->(d:Document {doc_id: $doc_id})
            RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name,
                   e.entity_type AS entity_type, r.mention_text AS mention_text,
                   r.confidence AS confidence
            ORDER BY r.confidence DESC
            """,
            doc_id=doc_id,
        )
        return [dict(record) for record in result]


def link_similar_documents(
    doc_id_a: str,
    doc_id_b: str,
    similarity: float,
) -> None:
    """Create a SIMILAR_TO relationship between two documents."""
    with get_session() as session:
        session.run(
            """
            MATCH (a:Document {doc_id: $doc_id_a})
            MATCH (b:Document {doc_id: $doc_id_b})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.similarity = $similarity
            """,
            doc_id_a=doc_id_a,
            doc_id_b=doc_id_b,
            similarity=similarity,
        )


def find_related_documents(
    doc_id: str,
    max_hops: int = 2,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Traverse the graph to find related documents within N hops.

    Finds documents related through shared entities, same classification,
    or explicit SIMILAR_TO links.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (start:Document {doc_id: $doc_id})
            CALL {
                WITH start
                MATCH (start)-[:CLASSIFIED_AS]->(dt:DocumentType)<-[:CLASSIFIED_AS]-(related:Document)
                WHERE related.doc_id <> start.doc_id
                RETURN related, 'same_type' AS relation, 1.0 AS relevance
                UNION
                WITH start
                MATCH (start)<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(related:Document)
                WHERE related.doc_id <> start.doc_id
                RETURN related, 'shared_entity' AS relation, 0.8 AS relevance
                UNION
                WITH start
                MATCH (start)-[:SIMILAR_TO]-(related:Document)
                WHERE related.doc_id <> start.doc_id
                RETURN related, 'similar' AS relation, 0.9 AS relevance
            }
            RETURN DISTINCT related.doc_id AS doc_id, related.filename AS filename,
                   collect(DISTINCT relation) AS relations,
                   max(relevance) AS max_relevance
            ORDER BY max_relevance DESC
            LIMIT $limit
            """,
            doc_id=doc_id,
            limit=limit,
        )
        return [dict(record) for record in result]
