# Open WebUI Functions for knightGPT

Files in this directory are **not** deployed via docker-compose or any file
mount. Open WebUI Functions live in Open WebUI's own database, so each file
here is installed by hand through the Admin Panel UI.

## `collection_id_filter.py`

**Problem it fixes:** Open WebUI's "Knowledge" collection-attachment feature
never reaches knightGPT's API as a field in the `/v1/chat/completions`
request body -- the body only ever contains `messages`, `model`, `stream`,
`tools`. Open WebUI knows internally which Knowledge collection is attached
(`body["files"]` / `body["metadata"]["files"]`), but that information is
dropped during Open WebUI's own flattening step before the plain
OpenAI-style payload is sent to an external backend like knightGPT.

This Filter function's `inlet()` hook runs *before* that flattening step, so
it can still see the attached-files list. It copies the first attached
collection's `id` onto a new top-level `collection_id` key, which (unlike the
`files` structure) is not a recognized/stripped field and should survive
flattening.

> **Unverified:** this has not been tested against a live Open WebUI
> instance. The exact shape of `body` / `__metadata__` inside `inlet()` is
> based on OWUI's documented Filter Functions interface, not empirical
> observation. Treat the install + verify steps below as the first real test
> of this.

### Install steps

1. Open Open WebUI as an admin user.
2. Go to **Admin Panel → Functions**.
3. Click **+** (add a new function).
4. Paste the entire contents of `collection_id_filter.py` into the editor.
5. Click **Save**. Open WebUI reads the `title` / `author` / `description` /
   `version` block at the top of the file as the function's metadata.
6. Enable the function:
   - Either toggle it on **globally**, or
   - Open the `knightgpt-rag` model's settings and enable this filter for
     that model specifically (preferred, so it doesn't run against
     unrelated backends/models).

### Verify it worked

1. In a chat using the `knightgpt-rag` model, attach a Knowledge collection
   (the existing OWUI "Knowledge" feature).
2. Send a message.
3. Check knightGPT's API logs for the existing debug line in
   `src/api/main.py`'s `/v1/chat/completions` handler -- it already dumps
   every top-level request body key:
   ```
   DEBUG full request body keys: [...]
   ```
4. Confirm `'collection_id'` now appears in that key list (it previously
   did not). The line above it,
   `DEBUG request_context: ... collection_id=...`, should also now show the
   attached collection's id instead of `None`.
5. If `collection_id` still does not appear, the candidate location this
   filter read the attached-files list from (`body["files"]`,
   `body["metadata"]["files"]`, or `__metadata__["files"]`) was wrong for
   this OWUI version -- inspect `__metadata__` / `body` directly (e.g. with
   a temporary `print()`/log in `inlet()`) to find the real location and
   update the filter.
