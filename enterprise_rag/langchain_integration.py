"""Optional LangChain orchestration over the same permissioned retrieval.

`pip install langchain-core`. The core pipeline (rag.py) does not depend on LangChain; this module
exposes the retriever and an LCEL chain for people who want to compose it with other LangChain
components. NOT executed in the sandbox this was written in (langchain not installed) - treat it
as untested until `tests/test_optional_backends.py` passes on your machine.

Permission rule: the retriever takes the Principal at construction and builds its Constraints from
the principal's groups, so a chain can never retrieve outside them.
"""

from __future__ import annotations

from typing import Any

from .access import Constraints, Principal
from .factory import Stack
from .index_manager import embedding_text
from .rag import RAGPipeline, validate_citations
from .retrieval import RetrievalOptions


def permissioned_retriever(stack: Stack, principal: Principal, options: RetrievalOptions | None = None):
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class PermissionedRetriever(BaseRetriever):
        stack: Any
        principal: Any
        options: Any = None

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
            constraints = Constraints(self.principal.groups, as_of=self.stack.rag.today)
            result = self.stack.retriever.retrieve(query, constraints, self.options)
            return [Document(page_content=e.chunk["text"], metadata={
                "chunk_id": e.chunk["chunk_id"], "source": e.chunk["source"], "version": e.chunk["version"],
                "title": e.chunk["title"], "score": e.score, "contextual_text": embedding_text(e.chunk)})
                for e in result.evidence]

    return PermissionedRetriever(stack=stack, principal=principal, options=options)


def build_chain(stack: Stack, principal: Principal, chat_model, options: RetrievalOptions | None = None):
    """question -> retrieve (permissioned) -> numbered evidence -> chat model -> citation-validated text."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    retriever = permissioned_retriever(stack, principal, options)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAGPipeline.SYSTEM),
        ("human", "Question: {question}\n\nEvidence:\n{evidence}")])

    def number(docs):
        return "\n\n".join(f"[E{i}] {d.metadata['title']} | version {d.metadata['version']} | {d.metadata['source']}\n"
                           f"{d.metadata['contextual_text']}" for i, d in enumerate(docs, 1))

    def validate(payload):
        text, count = payload["text"], len(payload["docs"])
        if text.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
            return {"answer": None, "abstained": True, "docs": payload["docs"]}
        valid, cleaned, problems = validate_citations(text, count)
        return {"answer": cleaned if valid else None, "abstained": not valid, "problems": problems,
                "docs": payload["docs"]}

    generate = prompt | chat_model | StrOutputParser()
    return (RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
            | RunnablePassthrough.assign(evidence=lambda x: number(x["docs"]))
            | RunnablePassthrough.assign(text=generate)
            | RunnableLambda(validate))
