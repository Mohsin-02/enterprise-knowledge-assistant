"""Streamlit front end.   streamlit run streamlit_app.py

Talks to the FastAPI service (ERAG_API_URL, default http://localhost:8000). Paste an API token in
the sidebar; what you can see is decided by the server from that token. Not executed in the
sandbox it was written in (streamlit not installed).
"""

import json
import os

import requests
import streamlit as st

API = os.environ.get("ERAG_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "What changes were made to the employee reimbursement policy between 2024 and 2026?",
    "What about the submission deadline?",
    "How many days of annual leave do I get?",
    "What does standard SEC-204 require?",
]

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")
st.title("Enterprise Knowledge Assistant")


def call(method: str, path: str, **kwargs):
    headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
    return requests.request(method, f"{API}{path}", headers=headers, timeout=120, **kwargs)


with st.sidebar:
    st.session_state["token"] = st.text_input("API token", type="password", value=st.session_state.get("token", ""))
    me = None
    if st.session_state["token"]:
        try:
            reply = call("GET", "/me")
            me = reply.json() if reply.ok else None
            if me is None:
                st.error("Token not accepted.")
        except requests.RequestException as exc:
            st.error(f"Cannot reach the API at {API}: {exc}")
    if me:
        st.success(f"Signed in as {me['user_id']}")
        if st.button("New conversation"):
            st.session_state.pop("session_id", None)
            st.session_state["messages"] = []
        st.subheader("Documents you can access")
        docs = call("GET", "/documents")
        for doc in docs.json() if docs.ok else []:
            st.caption(f"{doc['title']} - v{doc['version']} ({doc['source']})")
        if me["is_admin"]:
            st.subheader("Upload a document")
            upload = st.file_uploader("File", type=["pdf", "docx", "html", "htm", "xlsx", "txt", "md"])
            meta = st.text_area("Metadata (JSON)", json.dumps({
                "title": "New policy", "document_family": "new-policy", "version": "2026",
                "effective_from": "2026-01-01", "effective_to": None, "access_groups": ["employees"]}, indent=2))
            if upload and st.button("Ingest"):
                result = call("POST", "/documents", files={"file": (upload.name, upload.getvalue())}, data={"metadata": meta})
                st.write(result.json())

if not me:
    st.info("Enter an API token in the sidebar. Ask your administrator, or run `python manage.py seed-demo` locally.")
    st.stop()

st.session_state.setdefault("messages", [])
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            with st.expander(f"{citation['id']}: {citation['title']} (v{citation['version']})"):
                st.write(f"Source: {citation['source']}  |  Location: {citation.get('display', '')}")
                st.write(citation["text"])

st.caption("Try: " + " | ".join(EXAMPLES))
if question := st.chat_input("Ask about company policies and documents"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    reply = call("POST", "/ask", json={"question": question, "session_id": st.session_state.get("session_id")})
    if reply.ok:
        data = reply.json()
        st.session_state["session_id"] = data["session_id"]
        st.session_state["messages"].append({"role": "assistant", "content": data["answer"], "citations": data["citations"]})
    else:
        st.session_state["messages"].append({"role": "assistant", "content": f"Error: {reply.text}"})
    st.rerun()
