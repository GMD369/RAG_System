import os
from openai import BadRequestError
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utils import get_llm, get_vectorstore

load_dotenv()

SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def retrieve(query: str, score_threshold: float = SCORE_THRESHOLD, k: int = TOP_K) -> list:
    db = get_vectorstore()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} document(s) above threshold {score_threshold}:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata.get('source', 'unknown')} — {len(doc.page_content)} chars")
    return docs


def answer(query: str, docs: list) -> str:
    if not docs:
        return "I don't know based on the provided documents."

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict RAG assistant. Answer ONLY using the provided context. "
            "Do not use outside knowledge. If the answer is not in the context, reply exactly: "
            "'I don't know based on the provided documents.' "
            "Include a short source citation using the source file name(s).",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])

    try:
        response = (prompt | llm).invoke({"question": query, "context": context})
    except BadRequestError as exc:
        message = str(exc)
        if "Incorrect API key" in message or "invalid_api_key" in message:
            raise ValueError("Invalid API key. Check GROK_API_KEY in .env.") from exc
        if "Model not found" in message:
            raise ValueError(
                "Model not found. Update GROK_MODEL in .env or remove it to auto-select."
            ) from exc
        raise

    return response.content


def main():
    print("RAG Query System — type 'quit' to exit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        docs = retrieve(query)
        result = answer(query, docs)
        print(f"\n--- Answer ---\n{result}\n")


if __name__ == "__main__":
    main()
