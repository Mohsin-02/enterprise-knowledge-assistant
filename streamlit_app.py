"""Streamlit front end (dark theme).   streamlit run streamlit_app.py

Talks to the FastAPI service (ERAG_API_URL, default http://localhost:8000). Paste an API token in
the sidebar; what you can see is decided by the server from that token, never by this page.
Dark theme: .streamlit/config.toml (see config.toml next to this file).
"""

import html
import json
import os

import requests
import streamlit as st

API = os.environ.get("ERAG_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "What changes were made to the employee reimbursement policy between 2024 and 2026?",
    "How many days of annual leave do I get?",
    "What does standard SEC-204 require?",
    "What form must a manager submit to report misconduct?",
]

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")
st.markdown("""
<style>
.chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px; font-size:0.78rem;
       border:1px solid rgba(88,166,255,.45); background:rgba(88,166,255,.12); color:#9CC9FF;}
.chip.group {border-color:rgba(163,113,247,.45); background:rgba(163,113,247,.12); color:#D2B8FF;}
.refusal {border:1px solid rgba(248,81,73,.45); background:rgba(248,81,73,.08); border-radius:8px;
          padding:10px 14px; color:#FFB1AC;}
.quote {white-space:pre-wrap; border-left:3px solid rgba(88,166,255,.6); padding:6px 12px;
        background:rgba(110,118,129,.12); border-radius:4px; font-size:0.92rem;}
.meta {color:#8B949E; font-size:0.78rem;}
</style>
""", unsafe_allow_html=True)


class ApiUnavailable(Exception):
    pass


def call(method: str, path: str, token: str, **kwargs):
    try:
        return requests.request(method, f"{API}{path}", headers={"Authorization": f"Bearer {token}"},
                                timeout=180, **kwargs)
    except requests.RequestException as exc:
        raise ApiUnavailable(f"Cannot reach the API at {API}. Is it running? ({type(exc).__name__})") from None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_profile(token: str):
    reply = call("GET", "/me", token)
    return reply.json() if reply.ok else None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_documents(token: str):
    reply = call("GET", "/documents", token)
    return reply.json() if reply.ok else []


def error_text(reply) -> str:
    try:
        detail = reply.json().get("detail", reply.text)
    except ValueError:
        detail = reply.text
    return detail if isinstance(detail, str) else json.dumps(detail)


def chips(values, css="chip") -> str:
    return "".join(f'<span class="{css}">{html.escape(str(v))}</span>' for v in values)


def render_assistant(message: dict) -> None:
    if message.get("error"):
        st.error(message["content"])
        return
    if message.get("abstained"):
        st.markdown(f'<div class="refusal">{html.escape(message["content"])}</div>', unsafe_allow_html=True)
    else:
        versions = sorted({c["version"] for c in message.get("citations", [])})
        if versions:
            label = "Compared revisions" if message.get("comparison") else "Sources from"
            st.markdown(f'<span class="meta">{label}:</span> ' + chips(f"v{v}" for v in versions), unsafe_allow_html=True)
        st.markdown(message["content"])
    for note in message.get("warnings", []):
        st.caption(f"ℹ️ {note}")
    for c in message.get("citations", []):
        with st.expander(f"{c['id']} · {c['title']} · v{c['version']}"):
            st.markdown(f'<div class="meta">{html.escape(c["source"])} · {html.escape(c.get("display") or "")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="quote">{html.escape(c["text"])}</div>', unsafe_allow_html=True)
    if message.get("latency_ms") is not None:
        st.markdown(f'<span class="meta">{message["latency_ms"]:.0f} ms{" · cached" if message.get("cached") else ""}</span>',
                    unsafe_allow_html=True)


def render(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)


def ask(question: str, token: str) -> dict:
    """Send a question; start a fresh session once if the stored one is unknown to the server."""
    for attempt in range(2):
        reply = call("POST", "/ask", token, json={"question": question, "session_id": st.session_state.get("session_id")})
        if reply.status_code == 404 and attempt == 0:
            st.session_state.pop("session_id", None)
            continue
        break
    if not reply.ok:
        text = "Your token was not accepted." if reply.status_code == 401 else error_text(reply)
        return {"role": "assistant", "content": text, "error": True}
    data = reply.json()
    st.session_state["session_id"] = data["session_id"]
    return {"role": "assistant", "content": data["answer"], "abstained": data.get("abstained", False),
            "citations": data.get("citations", []), "warnings": data.get("warnings", []),
            "comparison": (data.get("analysis") or {}).get("comparison"), "latency_ms": data.get("latency_ms"),
            "cached": data.get("cached")}


# ---- sidebar -----------------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Sign in")
    token = st.text_input("API token", type="password", help="Demo tokens: data\\demo_tokens.json").strip()
    if token != st.session_state.get("active_token"):  # a different user: never show the old chat
        st.session_state.update(active_token=token, messages=[])
        st.session_state.pop("session_id", None)
        st.cache_data.clear()
    profile, api_error = None, None
    if token:
        try:
            profile = fetch_profile(token)
            if profile is None:
                api_error = "Token not accepted."
        except ApiUnavailable as exc:
            api_error = str(exc)
    if api_error:
        st.error(api_error)
    if profile:
        st.success(f"Signed in as **{profile['user_id']}**")
        st.markdown(chips(profile["groups"], "chip group"), unsafe_allow_html=True)
        if st.button("➕ New conversation", use_container_width=True):
            st.session_state.update(messages=[])
            st.session_state.pop("session_id", None)
        with st.expander("Documents you can access"):
            for doc in fetch_documents(token):
                groups = f" · {', '.join(doc['access_groups'])}" if doc.get("access_groups") else ""
                st.caption(f"**{doc['title']}** · v{doc['version']}{groups}")
        if profile["is_admin"]:
            with st.expander("Upload a document (admin)"):
                upload = st.file_uploader("File", type=["pdf", "docx", "html", "htm", "xlsx", "txt", "md"])
                meta = st.text_area("Metadata (JSON)", json.dumps({
                    "title": "New policy", "document_family": "new-policy", "version": "2026",
                    "effective_from": "2026-01-01", "effective_to": None, "access_groups": ["employees"]}, indent=2), height=220)
                if upload and st.button("Ingest"):
                    try:
                        result = call("POST", "/documents", token, files={"file": (upload.name, upload.getvalue())},
                                      data={"metadata": meta})
                        if result.ok:
                            st.success(f"Ingested {result.json().get('chunks')} passages.")
                            st.cache_data.clear()
                        else:
                            st.error(error_text(result))
                    except ApiUnavailable as exc:
                        st.error(str(exc))
    st.divider()
    st.caption("Answers use only documents your account may read. Permissions are enforced by the server.")

# ---- main --------------------------------------------------------------------------------------
st.title("Enterprise Knowledge Assistant")
if not profile:
    st.info("Paste an API token in the sidebar to start. Demo tokens are in `data\\demo_tokens.json` "
            "(run `python manage.py seed-demo` if the file is missing).")
    st.stop()

st.session_state.setdefault("messages", [])
for message in st.session_state["messages"]:
    render(message)

if not st.session_state["messages"]:
    st.subheader("Try asking")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        columns[i % 2].button(example, key=f"example-{i}", use_container_width=True,
                              on_click=st.session_state.__setitem__, args=("pending", example))

typed = st.chat_input("Ask about company policies and documents")
question = typed or st.session_state.pop("pending", None)
if question:
    user_message = {"role": "user", "content": question}
    st.session_state["messages"].append(user_message)
    render(user_message)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            try:
                answer = ask(question, token)
            except ApiUnavailable as exc:
                answer = {"role": "assistant", "content": str(exc), "error": True}
        render_assistant(answer)
    st.session_state["messages"].append(answer)
