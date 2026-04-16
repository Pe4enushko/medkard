from dataclasses import dataclass, field


@dataclass
class Doc:
    """A single ingested PDF chunk with its hypothetical queries and embeddings."""

    chunk: str
    file_id: str = ""                    # manifest ID (PDF filename stem)
    metadata: dict = field(default_factory=dict)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None

    # Hypothetical queries (reverse HyDE)
    fact_q: str | None = None
    procedure_q: str | None = None
    constraint_q: str | None = None

    # Embedding vectors — populated for writes, not fetched on reads.
    fact_q_embedding: list[float] | None = None
    procedure_q_embedding: list[float] | None = None
    constraint_q_embedding: list[float] | None = None
