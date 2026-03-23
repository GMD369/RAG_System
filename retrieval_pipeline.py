import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import BadRequestError, OpenAI
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


def resolve_grok_model(api_key, base_url, configured_model=None):
    """Resolve a valid Grok model. Use configured model if provided, else auto-select from xAI."""
    if configured_model:
        return configured_model

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        models_response = client.models.list()
    except BadRequestError as exc:
        message = str(exc)
        if "Incorrect API key" in message or "invalid_api_key" in message:
            raise ValueError(
                "Invalid xAI API key. Update XAI_API_KEY (or GROK_API_KEY) in .env with a valid key from https://console.x.ai."
            ) from exc
        raise
    available_models = [model.id for model in models_response.data]

    preferred_prefixes = ["grok-3", "grok-2", "grok", "xai"]
    for prefix in preferred_prefixes:
        for model_id in available_models:
            if model_id.startswith(prefix):
                return model_id

    if available_models:
        return available_models[0]

    raise ValueError("No models returned by xAI API for this key.")


def answer_with_grok(user_query, docs):
    """Generate an answer with Grok, strictly grounded in retrieved docs only."""
    context = "\n\n".join(
        [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs]
    )

    grok_api_key = os.getenv("GROK_API_KEY")
    if not grok_api_key:
        raise ValueError("Missing GROK_API_KEY. Set GROK_API_KEY in your .env.")

    configured_model = os.getenv("GROK_MODEL")
    configured_base_url = os.getenv("GROK_BASE_URL")

    if grok_api_key.startswith("gsk_"):
        grok_base_url = configured_base_url or "https://api.groq.com/openai/v1"
        grok_model = configured_model or "llama-3.3-70b-versatile"
    else:
        grok_base_url = configured_base_url or "https://api.x.ai/v1"
        grok_model = resolve_grok_model(grok_api_key, grok_base_url, configured_model)

    print(f"Using Grok model: {grok_model}")

    llm = ChatOpenAI(
        model=grok_model,
        api_key=grok_api_key,
        base_url=grok_base_url,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict RAG assistant. Answer ONLY using the provided context. "
                "Do not use outside knowledge. If the answer is not explicitly present in the context, "
                "reply exactly: 'I don't know based on the provided documents.' "
                "When you answer, include a short source citation using the source file name(s).",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}",
            ),
        ]
    )

    try:
        response = (prompt | llm).invoke({"question": user_query, "context": context})
    except BadRequestError as exc:
        message = str(exc)
        if "Incorrect API key" in message or "invalid_api_key" in message:
            raise ValueError(
                "Invalid xAI API key. Update XAI_API_KEY (or GROK_API_KEY) in .env with a valid key from https://console.x.ai."
            ) from exc
        if "Model not found" in message:
            raise ValueError(
                "Selected model is not available for your xAI account. "
                "If you set GROK_MODEL, update it to a valid one or remove it to auto-select."
            ) from exc
        raise

    return response.content


final_answer = answer_with_grok(query, relevant_docs)
print("--- Grok Answer (Context-Restricted) ---")
print(final_answer)


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"