# Recovery notes

Another Claude session was working in this folder at the same time, and twice files it had written
were overwritten by mistake. Nothing was recoverable from backups, so the files below were rebuilt.
If you have the earlier versions, prefer them and re-run all tests.

## Incident 1: access.py, embeddings.py, vector_store.py, fusion.py, semantic_chunking.py
- `access.py`, `embeddings.py`, `vector_store.py`: read back from surviving compiled copies and compared
  with the originals (1,620 access cases identical; hashing embedder identical; in-memory store identical on
  300 random searches). The Qdrant and sentence-transformer classes could not run offline.
- `fusion.py`, `semantic_chunking.py`: NOT the originals. New code written to the interface `retrieval.py`
  and `pipeline.py` call. `embeddings.py` also gained a `fingerprint` property (alias of `model_id`).

## Incident 2: db.py, service.py, settings.py
- Rebuilt from how `api.py`, `manage.py`, `factory.py` and `evaluate.py` call them, and from the existing
  `data/app.db` schema and rows (turn format, hashed tokens). The existing demo tokens still authenticate.
- NOT the originals: method details inside `service.py` (for example caching and rate limiting) and the
  `Settings` defaults are my reconstruction.
- `manage.py seed-demo` overwrote `data/demo_tokens.json` while testing; it was restored from the values
  printed earlier and checked against `data/app.db`.

## Evidence the rebuilt stack matches the other session's work
`python evaluate.py` reproduces the dev and test numbers recorded in `docs/EVALUATION.md`
(for example answer_correctness 0.955 on dev, 0 permission leaks), and `tests/test_service.py` plus
`tests/test_restored_modules.py` cover the rebuilt behaviour.
