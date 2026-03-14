"""Briefing bot email parser for KnightGPT.

Parses structured email briefings (from a daily/weekly briefing bot),
extracts DOIs, paper titles, and summaries for ingestion.

Expected briefing format (flexible — handles multiple styles):
- Plain text with DOIs inline
- Markdown with paper sections
- Structured JSON payload from bot API
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils import get_logger

logger = get_logger(__name__)

# DOI regex — matches 10.XXXX/... patterns
DOI_PATTERN = re.compile(
    r"(?:doi[:\s]*)?(?:https?://(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s,;)\]\"']+)",
    re.IGNORECASE,
)

# PubMed ID pattern
PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{6,9})", re.IGNORECASE)

# URL pattern for paper links
PAPER_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:"
    r"(?:ncbi\.nlm\.nih\.gov/pmc/articles/PMC\d+)"
    r"|(?:pubmed\.ncbi\.nlm\.nih\.gov/\d+)"
    r"|(?:(?:dx\.)?doi\.org/10\.\d{4,9}/[^\s]+)"
    r"|(?:biorxiv\.org/content/10\.\d{4,9}/[^\s]+)"
    r"|(?:medrxiv\.org/content/10\.\d{4,9}/[^\s]+)"
    r"|(?:nature\.com/articles/[^\s]+)"
    r"|(?:cell\.com/[^\s]+/fulltext/[^\s]+)"
    r"|(?:science\.org/doi/[^\s]+)"
    r")",
    re.IGNORECASE,
)


@dataclass
class ParsedPaperRef:
    """A paper reference extracted from a briefing."""

    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def has_identifier(self) -> bool:
        return bool(self.doi or self.pmid or self.url)


@dataclass
class ParsedBriefing:
    """A parsed briefing with extracted paper references."""

    raw_text: str
    papers: list[ParsedPaperRef]
    briefing_summary: Optional[str] = None
    source: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def parse_briefing_text(text: str, source: str = "email") -> ParsedBriefing:
    """
    Parse a briefing text and extract paper references.

    Handles multiple formats:
    - Free-form text with DOIs
    - Numbered/bulleted paper lists
    - Markdown sections with paper descriptions

    Args:
        text: Raw briefing text
        source: Source identifier (email, bot, manual)

    Returns:
        ParsedBriefing with extracted papers
    """
    if not text or not text.strip():
        return ParsedBriefing(raw_text=text or "", papers=[], source=source)

    papers = []
    seen_dois = set()

    # Extract all DOIs
    for match in DOI_PATTERN.finditer(text):
        doi = match.group(1).rstrip(".")
        if doi not in seen_dois:
            seen_dois.add(doi)
            # Try to find surrounding context for title/summary
            context = _extract_context(text, match.start(), match.end())
            papers.append(
                ParsedPaperRef(
                    doi=doi,
                    title=context.get("title"),
                    summary=context.get("summary"),
                )
            )

    # Extract PMIDs not already covered by DOIs
    for match in PMID_PATTERN.finditer(text):
        pmid = match.group(1)
        context = _extract_context(text, match.start(), match.end())
        papers.append(
            ParsedPaperRef(
                pmid=pmid,
                title=context.get("title"),
                summary=context.get("summary"),
            )
        )

    # Extract paper URLs not already covered
    for match in PAPER_URL_PATTERN.finditer(text):
        url = match.group(0)
        # Check if this URL contains a DOI we already extracted
        doi_in_url = DOI_PATTERN.search(url)
        if doi_in_url and doi_in_url.group(1) in seen_dois:
            continue
        context = _extract_context(text, match.start(), match.end())
        papers.append(
            ParsedPaperRef(
                url=url,
                title=context.get("title"),
                summary=context.get("summary"),
            )
        )

    # Extract a briefing-level summary (first paragraph or first few lines)
    lines = text.strip().split("\n")
    summary_lines = []
    for line in lines[:5]:
        line = line.strip()
        if line and not DOI_PATTERN.search(line) and not line.startswith(("http", "#")):
            summary_lines.append(line)
        if len(summary_lines) >= 3:
            break
    briefing_summary = " ".join(summary_lines) if summary_lines else None

    logger.info(
        f"Parsed briefing from {source}: {len(papers)} paper references found"
    )

    return ParsedBriefing(
        raw_text=text,
        papers=papers,
        briefing_summary=briefing_summary,
        source=source,
    )


def parse_briefing_json(data: dict) -> ParsedBriefing:
    """
    Parse a structured JSON briefing payload from a bot API.

    Expected format:
    {
        "text": "briefing summary text",
        "papers": [
            {"doi": "10.xxx/yyy", "title": "...", "summary": "..."},
            {"url": "https://...", "title": "..."},
        ],
        "source": "briefing-bot",
        "metadata": {...}
    }

    Args:
        data: JSON payload dict

    Returns:
        ParsedBriefing
    """
    papers = []
    for p in data.get("papers", []):
        papers.append(
            ParsedPaperRef(
                doi=p.get("doi"),
                pmid=p.get("pmid"),
                url=p.get("url"),
                title=p.get("title"),
                summary=p.get("summary"),
                metadata=p.get("metadata", {}),
            )
        )

    # Also extract DOIs from the text body if present
    text = data.get("text", "")
    if text:
        text_parsed = parse_briefing_text(text, source=data.get("source", "bot"))
        # Add any papers from text not already in the explicit list
        existing_dois = {p.doi for p in papers if p.doi}
        for tp in text_parsed.papers:
            if tp.doi and tp.doi not in existing_dois:
                papers.append(tp)

    return ParsedBriefing(
        raw_text=text,
        papers=papers,
        briefing_summary=data.get("text", "")[:500] if data.get("text") else None,
        source=data.get("source", "bot"),
        metadata=data.get("metadata", {}),
    )


def _extract_context(
    text: str, match_start: int, match_end: int
) -> dict:
    """Extract title and summary context around a DOI/PMID match."""
    result = {}

    # Look backwards for a title (bold text, heading, or preceding line)
    before = text[:match_start]
    lines_before = before.split("\n")

    # Check the line containing the match and the line before
    for line in reversed(lines_before[-3:]):
        line = line.strip()
        if not line:
            continue
        # Strip markdown formatting
        clean = re.sub(r"[*_#\[\]()]", "", line).strip()
        # Strip list markers
        clean = re.sub(r"^\d+[\.\)]\s*", "", clean)
        clean = re.sub(r"^[-•]\s*", "", clean)
        if clean and len(clean) > 10 and not DOI_PATTERN.search(clean):
            result["title"] = clean[:200]
            break

    # Look forward for a summary
    after = text[match_end:]
    lines_after = after.split("\n")
    summary_parts = []
    for line in lines_after[:4]:
        line = line.strip()
        if not line:
            if summary_parts:
                break
            continue
        if DOI_PATTERN.search(line) or PAPER_URL_PATTERN.search(line):
            break
        summary_parts.append(line)
    if summary_parts:
        result["summary"] = " ".join(summary_parts)[:500]

    return result
