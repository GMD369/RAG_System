import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import get_llm, get_vectorstore

load_dotenv()

SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def retrieve(query: str) -> list:
    db = get_vectorstore()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} relevant document(s).")
    return docs


def answer(query: str, docs: list) -> str:
    if not docs:
        return "I don't have enough information to answer based on the provided documents."

    context = "\n".join(f"- {doc.page_content}" for doc in docs)
    combined_input = (
        f"Based on the following documents, answer this question: {query}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer using only the information above. "
        "If the answer is not there, say 'I don't have enough information to answer that question based on the provided documents.'"
    )

    llm = get_llm()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    result = llm.invoke(messages)
    return result.content


def main():
    print("Answer Generation — type 'quit' to exit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        print(f"\nQuery: {query}")
        docs = retrieve(query)

        print("--- Context ---")
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")

        result = answer(query, docs)
        print(f"--- Answer ---\n{result}\n")


if __name__ == "__main__":
    main()
