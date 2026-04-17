from dataclasses import dataclass, field
import json, re

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

    def _format_chunk(self) -> str:
        """Return a human-readable string combining metadata and chunk content.

        Header line 1: Наименование | МКБ-10: <code> | Возрастная категория
        Header line 2: <section> | фрагмент <chunk_index>

        For table chunks: «Unnamed…» tokens are stripped and each JSON row
        is placed on its own line (content stays JSON).
        For text chunks: raw chunk text is used as-is.
        """
        content_type: str = self.metadata.get("content_type", "text")

        name: str | None = self.metadata.get("Наименование")
        mkb: str | None = self.metadata.get("МКБ-10")
        age_cat: str | None = self.metadata.get("Возрастная категория")
        section: str | None = self.metadata.get("section")
        chunk_idx = self.metadata.get("chunk_index")

        # Line 1: guideline identity
        id_parts: list[str] = []
        if name:
            id_parts.append(name)
        if mkb:
            id_parts.append(f"МКБ-10: {mkb}")
        if age_cat:
            id_parts.append(age_cat)

        # Line 2: section / position
        loc_parts: list[str] = []
        if section:
            loc_parts.append(section)
        if chunk_idx is not None:
            loc_parts.append(f"фрагмент {chunk_idx}")

        header_lines: list[str] = []
        if id_parts:
            header_lines.append(" | ".join(id_parts))
        if loc_parts:
            header_lines.append(" | ".join(loc_parts))

        header = "\n".join(header_lines)

        # Format content
        if content_type == "table":
            cleaned = re.sub(r"Unnamed\S*", " ", self.chunk)
            try:
                rows = json.loads(cleaned)
                body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            except (json.JSONDecodeError, TypeError):
                body = cleaned
        else:
            body = self.chunk

        if header:
            return f"[{header}]\n{body}"
        return body
