import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from utils import get_llm, get_vectorstore

load_dotenv()

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))


def rewrite_query(model, question: str, chat_history: list) -> str:
    if not chat_history:
        return question
    messages = [
        SystemMessage(
            content="Given the chat history, rewrite the new question to be standalone and searchable. "
                    "Return only the rewritten question, nothing else."
        ),
    ] + chat_history + [HumanMessage(content=f"New question: {question}")]
    result = model.invoke(messages)
    return result.content.strip()


def retrieve(search_query: str) -> list:
    db = get_vectorstore()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )
    return retriever.invoke(search_query)


def generate_answer(model, question: str, docs: list, chat_history: list) -> str:
    if not docs:
        return "I don't have enough information to answer based on the provided documents."

    context = "\n".join(f"- {doc.page_content}" for doc in docs)
    combined_input = (
        f"Based on the following documents, answer this question: {question}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer using only the information above. "
        "If it's not there, say 'I don't have enough information to answer that question based on the provided documents.'"
    )
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on provided documents and conversation history."
        ),
    ] + chat_history + [HumanMessage(content=combined_input)]

    result = model.invoke(messages)
    return result.content


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    model = get_llm()
    chat_history = []  # session-scoped, not global

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        search_question = rewrite_query(model, question, chat_history)
        if search_question != question:
            print(f"Searching for: {search_question}")

        docs = retrieve(search_question)
        print(f"Found {len(docs)} relevant document(s).")
        for i, doc in enumerate(docs, 1):
            preview = "\n".join(doc.page_content.split("\n")[:2])
            print(f"  Doc {i}: {preview}...")

        reply = generate_answer(model, question, docs, chat_history)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=reply))

        print(f"Answer: {reply}")


if __name__ == "__main__":
    start_chat()
