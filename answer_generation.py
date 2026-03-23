import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


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
query = "How much did Microsoft pay to acquire GitHub?"

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


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model (provider-aware, aligned with retrieval_pipeline.py)
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
    grok_model = configured_model or "grok-2-1212"

print(f"Using LLM model: {grok_model}")
model = ChatOpenAI(model=grok_model, api_key=grok_api_key, base_url=grok_base_url, temperature=0)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)