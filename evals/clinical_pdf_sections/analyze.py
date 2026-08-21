#!/usr/bin/env python3
"""Compare legacy and candidate section splitting across clinical PDFs.

This intentionally exercises only the text-section path.  It extracts page text
with PyMuPDF and applies the production text chunker, but skips table detection
and Tabula because neither participates in recognizing a bibliography heading.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
