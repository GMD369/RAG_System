import os
from openai import BadRequestError, OpenAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _resolve_xai_model(api_key: str, base_url: str, configured_model: str | None = None) -> str:
    if configured_model:
        return configured_model
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        models_response = client.models.list()
    except BadRequestError as exc:
        message = str(exc)
        if "Incorrect API key" in message or "invalid_api_key" in message:
            raise ValueError(
                "Invalid xAI API key. Update GROK_API_KEY in .env with a valid key from https://console.x.ai."
            ) from exc
        raise
    available = [m.id for m in models_response.data]
    for prefix in ["grok-3", "grok-2", "grok", "xai"]:
        for model_id in available:
            if model_id.startswith(prefix):
                return model_id
    if available:
        return available[0]
    raise ValueError("No models returned by xAI API for this key.")


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("Missing GROK_API_KEY. Set it in your .env file.")

    configured_model = os.getenv("GROK_MODEL")
    configured_base_url = os.getenv("GROK_BASE_URL")

    if api_key.startswith("gsk_"):
        base_url = configured_base_url or "https://api.groq.com/openai/v1"
        model = configured_model or "llama-3.3-70b-versatile"
    else:
        base_url = configured_base_url or "https://api.x.ai/v1"
        model = _resolve_xai_model(api_key, base_url, configured_model)

    print(f"Using LLM: {model}")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
