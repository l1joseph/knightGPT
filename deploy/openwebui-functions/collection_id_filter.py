"""
title: KnightGPT Collection ID Forwarder
author: l1joseph
description: >
    Extracts the id of the first attached Knowledge collection from Open
    WebUI's internal request state and injects it as a plain top-level
    `collection_id` key on the outgoing request body, so it survives Open
    WebUI's flattening step into the plain OpenAI-style payload sent to
    external backends (knightGPT's /v1/chat/completions). Without this,
    knightGPT's API never learns which Knowledge collection a chat has
    attached -- see src/api/request_context.py's body["collection_id"]
    fallback on the knightGPT side.
version: 0.1.0
"""

# UNVERIFIED: this file has not been run against a live Open WebUI
# instance from this environment (no reachable OWUI instance here). The
# exact shape of `body`, `__metadata__`, and `__user__` inside inlet() is
# based on Open WebUI's documented Filter Functions interface, not an
# empirical test. Before trusting this in production, paste it into a
# real OWUI instance, attach a Knowledge collection to a chat, send a
# message, and confirm `collection_id` shows up in knightGPT's existing
# "DEBUG full request body keys" log line (see src/api/main.py) where it
# didn't before. See deploy/openwebui-functions/README.md for the full
# install + verification steps.

from pydantic import BaseModel


class Filter:
    """Open WebUI Filter function: injects collection_id into the request
    body before Open WebUI flattens it into an OpenAI-style payload."""

    class Valves(BaseModel):
        """No configurable valves -- this filter has no settings yet."""

        pass

    def __init__(self) -> None:
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
    ) -> dict:
        """Extract the first attached collection's id and inject it into
        body["collection_id"] before Open WebUI forwards the request.

        Checks these candidate locations in order and uses the first one
        that is a non-empty list -- OWUI's internal layout for attached
        files is not something this codebase can verify locally, so all
        three plausible spots are tried defensively:
            - body.get("files")
            - body.get("metadata", {}).get("files")
            - (__metadata__ or {}).get("files")

        Each entry is expected to look like
        {"type": "collection", "id": "<knowledge_id>", "name": "..."}.
        Never raises: if none of the above are present, or no entry has
        type "collection", body is returned unchanged.

        Args:
            body: the outgoing request body Open WebUI is about to send.
            __user__: Open WebUI's current-user dict, if passed.
            __metadata__: Open WebUI's per-request metadata dict, if
                passed.

        Returns:
            body, with "collection_id" set if a collection attachment was
            found.
        """
        candidate_lists = [
            body.get("files"),
            (body.get("metadata") or {}).get("files"),
            (__metadata__ or {}).get("files"),
        ]

        for files in candidate_lists:
            if not files:
                continue
            for entry in files:
                if isinstance(entry, dict) and entry.get("type") == "collection":
                    collection_id = entry.get("id")
                    if collection_id:
                        body["collection_id"] = collection_id
                        return body

        return body
