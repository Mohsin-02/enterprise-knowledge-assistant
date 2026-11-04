"""Administration: python manage.py <command>   (run from the project root)

  build        Rebuild the search index from data/raw and register documents in the database.
  init-db      Create the database tables (SQLite by default, PostgreSQL via ERAG_DATABASE_URL).
  seed-demo    Create demo users and print their API tokens ONCE (also saved to data/demo_tokens.json).
  add-user     Create a user: --id alice --name "Alice" --groups employees,finance [--admin]
  set-groups   Replace a user's groups: --id alice --groups employees,hr
  audit        Show recent audit events.
"""

import argparse
import json
from pathlib import Path

from enterprise_rag.db import Database
from enterprise_rag.factory import build_stack
from enterprise_rag.metadata import load_registry
from enterprise_rag.service import KnowledgeService
from enterprise_rag.settings import Settings

DEMO = [
    ("alice", "Alice (employee)", ["employees"], False),
    ("bob", "Bob (finance)", ["employees", "finance"], False),
    ("carol", "Carol (HR)", ["employees", "hr"], False),
    ("dave", "Dave (IT and security)", ["employees", "it", "security"], False),
    ("admin", "Administrator", None, True),  # None = every group found in the registry
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "init-db", "seed-demo", "add-user", "set-groups", "audit"])
    parser.add_argument("--id")
    parser.add_argument("--name")
    parser.add_argument("--groups", default="")
    parser.add_argument("--admin", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    db = Database(settings.database_url)
    db.init_schema()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    if args.command == "init-db":
        print(f"Database ready: {settings.database_url}")
    elif args.command == "build":
        stack = build_stack(settings, rebuild=True)
        KnowledgeService(stack, db).sync_document_records()
        docs = stack.index.documents()
        print(f"Indexed {len(docs)} documents, {len(stack.index.chunks)} chunks "
              f"(embedder={stack.embedder.fingerprint}, strategy={settings.chunk_strategy}).")
    elif args.command == "seed-demo":
        registry = load_registry(settings.registry) or {}
        every = sorted({g for record in registry.values() for g in record["access_groups"]} | {"employees"})
        tokens = {}
        for user_id, name, user_groups, admin in DEMO:
            token = db.upsert_user(user_id, name, every if user_groups is None else user_groups, is_admin=admin)
            if token:
                tokens[user_id] = token
        if tokens:
            path = Path("data/demo_tokens.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
            print("Demo tokens (shown once; also saved to data/demo_tokens.json - DEVELOPMENT ONLY):")
            for user_id, token in tokens.items():
                print(f"  {user_id:6s} {token}")
        else:
            print("Demo users already exist; existing tokens are unchanged (only hashes are stored).")
    elif args.command == "add-user":
        if not args.id or not args.name:
            parser.error("add-user needs --id and --name")
        token = db.upsert_user(args.id, args.name, groups, is_admin=args.admin)
        print(f"User {args.id} saved." + (f" Token (shown once): {token}" if token else ""))
    elif args.command == "set-groups":
        if not args.id:
            parser.error("set-groups needs --id")
        db.set_groups(args.id, groups)
        db.audit(None, "user.groups", {"user_id": args.id, "groups": groups})
        print(f"Groups for {args.id}: {groups}")
    elif args.command == "audit":
        for event in db.audit_events():
            print(event["created_at"], event["user_id"], event["event"], event["detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
