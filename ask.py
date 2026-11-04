"""Ask questions from the command line (no API server needed).

    python manage.py seed-demo                       # once: creates demo users, saves data/demo_tokens.json
    python ask.py --user alice "How many days of annual leave do I get?"
    python ask.py --user bob                         # interactive; remembers follow-ups
    python ask.py --token <token> "question"
"""

import argparse
import json
from pathlib import Path
import sys

from enterprise_rag.db import Database
from enterprise_rag.display import format_citation
from enterprise_rag.factory import build_stack
from enterprise_rag.service import KnowledgeService, NotFound
from enterprise_rag.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--user", help="demo user id from data/demo_tokens.json (alice, bob, carol, dave, admin)")
    parser.add_argument("--token", help="API token (overrides --user)")
    parser.add_argument("--session", help="continue a session id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    token = args.token
    if not token and args.user:
        path = Path("data/demo_tokens.json")
        token = json.loads(path.read_text(encoding="utf-8")).get(args.user) if path.exists() else None
    settings = Settings.from_env()
    db = Database(settings.database_url)
    db.init_schema()
    user = db.authenticate(token)
    if user is None:
        print("Sign-in failed. Run `python manage.py seed-demo`, then use --user alice (or pass --token).", file=sys.stderr)
        return 1
    service = KnowledgeService(build_stack(settings), db)
    service.sync_document_records()
    print(f"Signed in as {user.user_id} (groups: {', '.join(sorted(user.groups))})", file=sys.stderr)
    session, pending = args.session, ([args.question] if args.question else None)
    while True:
        if pending is None:
            try:
                question = input("\nQuestion (blank to quit): ").strip()
            except EOFError:
                break
        elif pending:
            question = pending.pop()
        else:
            break
        if not question:
            break
        try:
            result = service.ask(user, question, session)
        except (ValueError, NotFound) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        session = result["session_id"]
        print("\n" + result["answer"])
        if result["citations"]:
            print("\nSources:")
            for citation in result["citations"]:
                print("  " + format_citation(citation))
        for warning in result["warnings"]:
            print(f"  note: {warning}")
        if args.verbose:
            print(json.dumps({k: result[k] for k in ("analysis", "diagnostics", "latency_ms", "cached")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
