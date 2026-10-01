"""Per-request identity and collection scope for knightGPT's API.

Built once per /v1/chat/completions and /api/v1/agent/chat request (see
src/api/main.py) from trusted reverse-proxy headers and the OpenAI-style
`files` array Open WebUI sends when a Knowledge collection is attached to
a chat. Threaded through AgentOrchestrator.run() into ingest_paper/
search_corpus as a keyword-only argument the model's JSON tool-call
arguments can never populate -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.

collection_id stays `str | None` at this layer (`None` = "no collection
attached" is the natural Python idiom); every translation into a SQL
column value or pgGraph session GUC maps `None` to the literal string
'global' at that specific boundary (src/retrieval/hybrid_retriever.py's
resolve_collection_id()) -- never here, and never passed through as
Python None to a query parameter.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RequestContext:
    """Identity + collection scope for one API request.

    email: the caller's address from a trusted proxy header, or None if
        neither header was present (e.g. a request that somehow reached
        the API outside both trusted proxy paths).
    is_admin: True iff email is non-None and present (case-insensitively)
        in settings.api.admin_email_set.
    collection_id: the attached Open WebUI Knowledge collection's id, or
        None if no collection is attached to this chat.
    """

    email: str | None = None
    is_admin: bool = False
    collection_id: str | None = None


def _first_matching_header(headers: Mapping[str, str], names: list[str]) -> str | None:
    """Case-insensitive lookup of the first present header among `names`,
    in priority order. Works for both a plain dict (tests) and Starlette's
    Headers (already case-insensitive, but this does its own lowercasing
    too so it never depends on that)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return None


def _first_collection_id(body: dict) -> str | None:
    """First {"type": "collection", "id": ...} entry in body["files"], if
    any -- the shape Open WebUI sends when a Knowledge collection is
    attached to a chat. See the spec's Components section."""
    for entry in body.get("files") or []:
        if isinstance(entry, dict) and entry.get("type") == "collection":
            collection_id = entry.get("id")
            if collection_id:
                return collection_id
    return None


def build_request_context(
    headers: Mapping[str, str],
    body: dict,
    admin_emails: set[str],
) -> RequestContext:
    """Build a RequestContext from one request's headers + parsed JSON body.

    Args:
        headers: the request's headers (any case-insensitive-or-not
            str-keyed Mapping -- e.g. starlette.datastructures.Headers or
            a plain dict).
        body: the request's parsed JSON body.
        admin_emails: settings.api.admin_email_set -- already lowercased.
    """
    email = _first_matching_header(
        headers, ["X-OpenWebUI-User-Email", "X-Auth-Request-Email"]
    )
    is_admin = email is not None and email.lower() in admin_emails
    collection_id = _first_collection_id(body)
    return RequestContext(email=email, is_admin=is_admin, collection_id=collection_id)
