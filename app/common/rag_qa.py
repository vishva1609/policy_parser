"""
RAG question answering over an indexed embedder.

Mirrors common/openapi.py:get_response_from_query from archit1012/qa-bot-llm,
adapted for EmbeddingPipeline + Mistral instead of FAISS + ChatOpenAI.
"""
from typing import Any, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_client import MistralConfig


def get_response_from_query(
    embedder: Any,
    query: str,
    llm_config: MistralConfig,
    depth: int = 4,
) -> Tuple[str, List[str]]:
    """
    Retrieve top-k chunks for ``query``, then answer with the shared LLM.

    Args:
        embedder:   EmbeddingPipeline (or any object with ``search(query, n_results=...)``).
        query:      User question.
        llm_config: Injected MistralConfig (same role as ``chat`` in the reference repo).
        depth:      Number of chunks to retrieve (reference default k=4).

    Returns:
        (answer_string, list_of_retrieved_chunk_texts)
    """
    raw = embedder.search(query, n_results=depth)
    chunks: List[str] = []
    if raw.get("documents") and raw["documents"][0]:
        chunks = list(raw["documents"][0])
    context = " ".join(chunks)

    system = (
        "You are a helpful assistant that answers questions using only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Rules:\n"
        "- Only use factual information from the context.\n"
        "- If you do not have enough information to answer, say that you don't have "
        "enough information to answer the question.\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Answer the following question: {question}"),
        ]
    )
    chain = prompt | llm_config.llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": query})
    text = (response or "").replace("\n", " ").strip()
    return text, chunks
