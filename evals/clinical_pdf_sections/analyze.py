#!/usr/bin/env python3
"""Compare legacy and candidate section splitting across clinical PDFs.

This intentionally exercises only the text-section path.  It extracts page text
with PyMuPDF and applies the production text chunker, but skips table detection
and Tabula because neither participates in recognizing a bibliography heading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from RAG.ingestion.data_loader import _split_into_sections, _text_splitter


_LEGACY_SECTION_PATTERN = re.compile(
    r"^("
    r"\d+\.\d+(?:\.\d+)?\s+"
    r"(?=[^\n]*[A-Za-zА-Яа-я])"
    r"(?![^\n]*\.{2,})"
    r".*?"
    r")(?=^\d+\.\d+(?:\.\d+)?\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_LEGACY_TITLE_PATTERN = re.compile(
    r"^("
    r"\d+\.\d+(?:\.\d+)?\s+"
    r"(?=[^\n]*[A-Za-zА-Яа-я])"
    r"(?![^\n]*\.{2,})"
    r"[^\n]+"
    r")",
    re.MULTILINE,
)
_BIBLIOGRAPHY_TITLES = {
    "список литературы",
    "список использованной литературы",
    "список использованных источников",
    "библиографический список",
    "литература",
}
_CRITERIA_PHRASE = "критерии оценки качества"


@dataclass(frozen=True)
class SectionStat:
    title: str | None
    chars: int
    chunks: int


@dataclass(frozen=True)
class DocumentStat:
    filename: str
    file_size: int
    sha256: str
    legacy: tuple[SectionStat, ...]
    candidate: tuple[SectionStat, ...]
    literature_lines: tuple[str, ...]
    criteria_lines: tuple[str, ...]


def _legacy_split(text: str) -> list[tuple[str | None, str]]:
    matches = _LEGACY_SECTION_PATTERN.findall(text)
    if not matches:
        return [(None, text)]

    result: list[tuple[str | None, str]] = []
    for section_text in matches:
        section_text = section_text.strip()
        if not section_text:
            continue
        title_match = _LEGACY_TITLE_PATTERN.match(section_text)
        title = title_match.group(1).strip() if title_match else None
        result.append((title, section_text))
    return result


def _extract_text(path: Path) -> str:
    with fitz.open(path) as document:
        parts = [page.get_text("text").strip() for page in document]
    text = "\n".join(part for part in parts if part).strip()

    abbreviation_heading = "Список сокращений"
    first = text.find(abbreviation_heading)
    if first != -1:
        second = text.find(abbreviation_heading, first + len(abbreviation_heading))
        if second != -1:
            text = text[second:].strip()
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _section_stats(sections: list[tuple[str | None, str]]) -> tuple[SectionStat, ...]:
    return tuple(
        SectionStat(
            title=title,
            chars=len(text),
            chunks=len(_text_splitter.split_text(text)),
        )
        for title, text in sections
    )


def _analyze_one(path_text: str) -> DocumentStat:
    path = Path(path_text)
    text = _extract_text(path)
    return DocumentStat(
        filename=path.name,
        file_size=path.stat().st_size,
        sha256=_sha256(path),
        legacy=_section_stats(_legacy_split(text)),
        candidate=_section_stats(_split_into_sections(text)),
        literature_lines=tuple(
            line
            for raw_line in text.splitlines()
            if (line := " ".join(raw_line.split()))
            and len(line) <= 120
            and "литератур" in line.casefold()
            and ".." not in line
        ),
        criteria_lines=tuple(
            line
            for raw_line in text.splitlines()
            if (line := " ".join(raw_line.split()))
            and len(line) <= 180
            and _CRITERIA_PHRASE in line.casefold()
            and ".." not in line
        ),
    )


def _is_criteria(section: SectionStat) -> bool:
    return bool(section.title and _CRITERIA_PHRASE in section.title.casefold())


def _is_bibliography(section: SectionStat) -> bool:
    if not section.title:
        return False
    normalized = re.sub(
        r"^(?:(?:[ivxlcdm]+|\d+)\.[ \t]+)?",
        "",
        section.title.casefold(),
    ).rstrip(".:")
    return normalized in _BIBLIOGRAPHY_TITLES


def _is_appendix(section: SectionStat) -> bool:
    return bool(section.title and section.title.casefold().startswith("приложени"))


def _compact_title(title: str | None) -> str:
    return " ".join(title.split()) if title else "(без раздела)"


def _section_kind(section: SectionStat) -> str:
    if section.title is None:
        return "unsectioned"
    if _is_bibliography(section):
        return "bibliography"
    if _is_criteria(section):
        return "criteria"
    if _is_appendix(section):
        return "appendix"
    if re.match(r"\d+\.\d+(?:\.\d+)?\s+", section.title):
        return "numbered"
    return "other"


_CANONICAL_PREFIXES = (
    re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+"),
    re.compile(r"^[ivxlcdm]+[.)]?\s+", re.IGNORECASE),
    re.compile(r"^таблица\s+\d+(?:\.\d+)*\s*[-–—.:]?\s*", re.IGNORECASE),
    re.compile(
        r"^приложение\s+[а-яa-z]\d*(?:\s*[-–—]\s*[а-яa-z0-9]+)?"
        r"\s*[.:-]?\s*",
        re.IGNORECASE,
    ),
)


def _canonical_title(title: str | None) -> str:
    """Return an exploratory key; exact titles remain the authoritative baseline."""
    value = _compact_title(title)
    if title is None:
        return value
    for pattern in _CANONICAL_PREFIXES:
        value = pattern.sub("", value, count=1)
    value = re.sub(r"\s+", " ", value).strip(" .:;,-–—").casefold()
    return value or _compact_title(title).casefold()


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return {row["ID"]: row for row in csv.DictReader(source)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _guideline_label(document_id: str, manifest_row: dict[str, str]) -> str:
    name = manifest_row.get("Наименование", "").strip()
    return f"{document_id} — {name}" if name else document_id


def _summary_rows(
    details: list[dict[str, object]], key_field: str
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in details:
        grouped.setdefault(str(row[key_field]), []).append(row)

    result: list[dict[str, object]] = []
    for title, rows in grouped.items():
        chunks = [int(row["chunks"]) for row in rows]
        guideline_labels = sorted(
            {
                str(row["guideline_label"])
                for row in rows
            },
            key=str.casefold,
        )
        kinds = sorted({str(row["section_kind"]) for row in rows})
        result.append(
            {
                key_field: title,
                "section_kinds": " | ".join(kinds),
                "section_occurrences": len(rows),
                "guideline_count": len(guideline_labels),
                "median_chunks": statistics.median(chunks),
                "mean_chunks": f"{statistics.fmean(chunks):.3f}",
                "min_chunks": min(chunks),
                "max_chunks": max(chunks),
                "total_chunks": sum(chunks),
                "guidelines": " | ".join(guideline_labels),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -int(row["guideline_count"]),
            str(row[key_field]).casefold(),
        ),
    )


def _write_baseline(
    output_dir: Path,
    stats: list[DocumentStat],
    failures: list[tuple[str, str]],
    manifest_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    details: list[dict[str, object]] = []
    corpus_rows: list[dict[str, object]] = []
    for stat in sorted(stats, key=lambda item: item.filename.casefold()):
        document_id = Path(stat.filename).stem
        manifest_row = manifest.get(document_id, {})
        label = _guideline_label(document_id, manifest_row)
        corpus_rows.append(
            {
                "guideline_id": document_id,
                "filename": stat.filename,
                "guideline_name": manifest_row.get("Наименование", ""),
                "icd10": manifest_row.get("МКБ-10", ""),
                "age_category": manifest_row.get("Возрастная категория", ""),
                "file_size_bytes": stat.file_size,
                "sha256": stat.sha256,
                "candidate_section_count": len(stat.candidate),
                "candidate_chunk_count": sum(section.chunks for section in stat.candidate),
            }
        )
        for section_order, section in enumerate(stat.candidate, start=1):
            exact_title = _compact_title(section.title)
            details.append(
                {
                    "guideline_id": document_id,
                    "filename": stat.filename,
                    "guideline_name": manifest_row.get("Наименование", ""),
                    "guideline_label": label,
                    "icd10": manifest_row.get("МКБ-10", ""),
                    "age_category": manifest_row.get("Возрастная категория", ""),
                    "section_order": section_order,
                    "section_title": exact_title,
                    "canonical_title": _canonical_title(section.title),
                    "section_kind": _section_kind(section),
                    "chunks": section.chunks,
                    "chars": section.chars,
                }
            )

    detail_fields = [
        "guideline_id",
        "filename",
        "guideline_name",
        "guideline_label",
        "icd10",
        "age_category",
        "section_order",
        "section_title",
        "canonical_title",
        "section_kind",
        "chunks",
        "chars",
    ]
    _write_csv(output_dir / "guideline-sections.csv", detail_fields, details)
    corpus_fields = list(corpus_rows[0]) if corpus_rows else []
    _write_csv(output_dir / "corpus.csv", corpus_fields, corpus_rows)

    summary_fields = [
        "section_title",
        "section_kinds",
        "section_occurrences",
        "guideline_count",
        "median_chunks",
        "mean_chunks",
        "min_chunks",
        "max_chunks",
        "total_chunks",
        "guidelines",
    ]
    exact_rows = _summary_rows(details, "section_title")
    _write_csv(output_dir / "sections-exact.csv", summary_fields, exact_rows)
    canonical_fields = summary_fields.copy()
    canonical_fields[0] = "canonical_title"
    canonical_rows = _summary_rows(details, "canonical_title")
    _write_csv(output_dir / "sections-canonical.csv", canonical_fields, canonical_rows)

    fingerprint = hashlib.sha256()
    for row in corpus_rows:
        fingerprint.update(f"{row['filename']}\0{row['sha256']}\n".encode())
    try:
        manifest_label = str(manifest_path.relative_to(ROOT))
    except ValueError:
        manifest_label = str(manifest_path)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": fingerprint.hexdigest(),
        "documents_ok": len(stats),
        "documents_failed": len(failures),
        "section_occurrences": len(details),
        "exact_section_titles": len(exact_rows),
        "canonical_section_titles": len(canonical_rows),
        "manifest": manifest_label,
        "failures": [{"filename": name, "error": error} for name, error in failures],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = (
        "# Baseline разбивки клинических рекомендаций\n\n"
        "Этот каталог — воспроизводимый снимок текстового section-parser на корпусе "
        f"из **{len(stats)} PDF**. Корпус идентифицируется SHA-256 "
        f"`{metadata['corpus_sha256']}`.\n\n"
        "## Файлы\n\n"
        "- `sections-exact.csv` — точный список заголовков после схлопывания "
        "пробелов: медиана, среднее, диапазон и список КР. Это основной baseline.\n"
        "- `sections-canonical.csv` — та же сводка после эвристического удаления "
        "нумерации (`8.`, `8.3`, `XIII`, `Таблица N`, `Приложение АN`). "
        "Она удобна для исследования, но не заменяет точную таблицу.\n"
        "- `guideline-sections.csv` — длинная таблица «КР → секция»: порядок, "
        "точное и каноническое название, тип, число чанков и символов.\n"
        "- `corpus.csv` — состав корпуса, метаданные КР, размер и SHA-256 каждого PDF.\n"
        "- `metadata.json` — дата генерации, fingerprint корпуса и контрольные количества строк.\n\n"
        "Медиана и среднее считаются по **вхождениям секций**, а не по сумме всего "
        "документа. `section_occurrences` явно показывает повторные одноимённые "
        "таблицы внутри одной КР, `guideline_count` — число разных КР.\n\n"
        "Это baseline текстового пути PyMuPDF + production text splitter. Поиск "
        "таблиц и Tabula намеренно не запускаются; их вклад надо измерять отдельно. "
        "Для сравнения нового парсера нужен тот же `corpus_sha256` и нулевой список "
        "ошибок в `metadata.json`.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"baseline written to {output_dir}", file=sys.stderr)


def _bucket(chunks: int) -> str:
    if chunks == 1:
        return "1"
    if chunks <= 5:
        return "2-5"
    if chunks <= 10:
        return "6-10"
    if chunks <= 25:
        return "11-25"
    if chunks <= 50:
        return "26-50"
    return ">50"


def _print_distribution(label: str, sections: list[SectionStat]) -> None:
    counts = Counter(_bucket(section.chunks) for section in sections)
    buckets = ("1", "2-5", "6-10", "11-25", "26-50", ">50")
    rendered = " ".join(f"{bucket}={counts[bucket]}" for bucket in buckets)
    print(f"{label}: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write reproducible per-section baseline CSV files to this directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "resources" / "manifest.csv",
        help="clinical guideline manifest used to resolve PDF IDs to readable names",
    )
    args = parser.parse_args()

    paths = sorted(args.directory.glob("*.pdf"))
    if not paths:
        parser.error(f"no PDF files found in {args.directory}")

    stats: list[DocumentStat] = []
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [(path, executor.submit(_analyze_one, str(path))) for path in paths]
        for index, (path, future) in enumerate(futures, start=1):
            try:
                stats.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report malformed PDFs and continue
                failures.append((path.name, f"{type(exc).__name__}: {exc}"))
            if index % 50 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}", file=sys.stderr)

    legacy_sections = [section for stat in stats for section in stat.legacy]
    candidate_sections = [section for stat in stats for section in stat.candidate]
    legacy_criteria = [section for section in legacy_sections if _is_criteria(section)]
    candidate_criteria = [section for section in candidate_sections if _is_criteria(section)]

    if args.output_dir:
        _write_baseline(args.output_dir, stats, failures, args.manifest)

    print(f"documents: total={len(paths)} ok={len(stats)} failed={len(failures)}")
    print(f"sections: legacy={len(legacy_sections)} candidate={len(candidate_sections)}")
    _print_distribution("all section chunk counts, legacy", legacy_sections)
    _print_distribution("all section chunk counts, candidate", candidate_sections)
    print(
        "criteria: "
        f"legacy_sections={len(legacy_criteria)} "
        f"legacy_chunks={sum(section.chunks for section in legacy_criteria)} "
        f"legacy_chars={sum(section.chars for section in legacy_criteria)} "
        f"candidate_sections={len(candidate_criteria)} "
        f"candidate_chunks={sum(section.chunks for section in candidate_criteria)} "
        f"candidate_chars={sum(section.chars for section in candidate_criteria)}"
    )
    _print_distribution("criteria chunk counts, legacy", legacy_criteria)
    _print_distribution("criteria chunk counts, candidate", candidate_criteria)
    criteria_by_document: list[tuple[int, int, str, int]] = []
    criteria_entries: list[tuple[int, int, str, str]] = []
    for stat in stats:
        document_criteria = [
            section for section in stat.candidate if _is_criteria(section)
        ]
        if not document_criteria:
            continue
        criteria_by_document.append(
            (
                sum(section.chunks for section in document_criteria),
                sum(section.chars for section in document_criteria),
                stat.filename,
                len(document_criteria),
            )
        )
        criteria_entries.extend(
            (
                section.chunks,
                section.chars,
                stat.filename,
                section.title or "—",
            )
            for section in document_criteria
        )
    print(
        "criteria documents: "
        f"count={len(criteria_by_document)} "
        f"max_chunks={max((row[0] for row in criteria_by_document), default=0)}"
    )
    print("largest criteria totals by document (chunks, sections, chars):")
    for chunks, chars, filename, section_count in sorted(
        criteria_by_document, reverse=True
    )[: args.top]:
        print(f"  {filename}: {chunks}, {section_count}, {chars}")
    print("largest individual criteria sections (chunks, chars):")
    for chunks, chars, filename, title in sorted(criteria_entries, reverse=True)[
        : args.top
    ]:
        print(f"  {filename}: {title!r}: {chunks}, {chars}")

    bibliography_sections = [
        section for section in candidate_sections if _is_bibliography(section)
    ]
    bibliography_titles = Counter(
        section.title.casefold() for section in bibliography_sections if section.title
    )
    print(
        "bibliography boundaries: "
        f"sections={len(bibliography_sections)} "
        f"chunks={sum(section.chunks for section in bibliography_sections)} "
        + " ".join(
            f"{title!r}={count}" for title, count in bibliography_titles.most_common()
        )
    )
    appendix_sections = [section for section in candidate_sections if _is_appendix(section)]
    print(
        "appendix boundaries: "
        f"sections={len(appendix_sections)} "
        f"chunks={sum(section.chunks for section in appendix_sections)}"
    )
    literature_lines = Counter(
        line.casefold() for stat in stats for line in stat.literature_lines
    )
    print("most common short lines containing 'литератур':")
    for line, count in literature_lines.most_common(args.top):
        print(f"  {count}: {line!r}")
    criteria_lines = Counter(line.casefold() for stat in stats for line in stat.criteria_lines)
    print("most common short lines containing 'критерии оценки качества':")
    for line, count in criteria_lines.most_common(args.top):
        print(f"  {count}: {line!r}")

    reductions: list[tuple[int, str, str, int, int, int]] = []
    for stat in stats:
        old_by_title = {section.title: section for section in stat.legacy}
        for index, section in enumerate(stat.candidate):
            if not _is_bibliography(section) or index == 0:
                continue
            previous = stat.candidate[index - 1]
            legacy_previous = old_by_title.get(previous.title)
            if legacy_previous is None:
                continue
            reduction = legacy_previous.chunks - previous.chunks
            if reduction > 0:
                reductions.append(
                    (
                        reduction,
                        stat.filename,
                        previous.title or "—",
                        legacy_previous.chunks,
                        previous.chunks,
                        section.chunks,
                    )
                )

    print(
        "directly comparable shortened pre-bibliography sections: "
        f"{len(reductions)}"
    )
    print("largest reductions (old -> new, bibliography):")
    for reduction, filename, title, old, new, bibliography in sorted(
        reductions, reverse=True
    )[: args.top]:
        print(
            f"  {filename}: {title!r}: {old} -> {new} "
            f"(-{reduction}), bibliography={bibliography}"
        )

    criteria_reductions: list[tuple[int, str, str, int, int]] = []
    for stat in stats:
        candidate_by_title = {section.title: section for section in stat.candidate}
        for legacy_section in stat.legacy:
            if not _is_criteria(legacy_section):
                continue
            candidate_section = candidate_by_title.get(legacy_section.title)
            if candidate_section is None:
                continue
            reduction = legacy_section.chunks - candidate_section.chunks
            if reduction > 0:
                criteria_reductions.append(
                    (
                        reduction,
                        stat.filename,
                        legacy_section.title or "—",
                        legacy_section.chunks,
                        candidate_section.chunks,
                    )
                )

    print(f"shortened criteria sections: {len(criteria_reductions)}")
    print("largest criteria reductions (old -> new):")
    for reduction, filename, title, old, new in sorted(
        criteria_reductions, reverse=True
    )[: args.top]:
        print(f"  {filename}: {title!r}: {old} -> {new} (-{reduction})")

    title_differences: list[tuple[str, list[str], list[str]]] = []
    missing_numbered = 0
    added_numbered = 0
    for stat in stats:
        legacy_titles = Counter(
            section.title
            for section in stat.legacy
            if section.title and re.match(r"\d+\.\d+(?:\.\d+)?\s+", section.title)
        )
        candidate_titles = Counter(
            section.title
            for section in stat.candidate
            if section.title and re.match(r"\d+\.\d+(?:\.\d+)?\s+", section.title)
        )
        missing = list((legacy_titles - candidate_titles).elements())
        added = list((candidate_titles - legacy_titles).elements())
        if missing or added:
            missing_numbered += len(missing)
            added_numbered += len(added)
            title_differences.append((stat.filename, missing, added))

    print(
        "numbered-title parity: "
        f"missing_from_candidate={missing_numbered} "
        f"added_by_candidate={added_numbered} "
        f"affected_documents={len(title_differences)}"
    )
    for filename, missing, added in title_differences[: args.top]:
        print(f"  {filename}: missing={missing!r} added={added!r}")

    if failures:
        print("failures:")
        for filename, error in failures[: args.top]:
            print(f"  {filename}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
