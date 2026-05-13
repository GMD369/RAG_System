import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from utils import get_embeddings, PERSIST_DIR

load_dotenv()


def load_documents(docs_path: str = "docs") -> list:
    print(f"Loading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' not found. Create it and add your documents.")

    txt_loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    )
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = txt_loader.load() + pdf_loader.load()

    if not documents:
        raise FileNotFoundError(f"No .txt or .pdf files found in '{docs_path}'.")

    print(f"Loaded {len(documents)} document(s):")
    for doc in documents[:5]:
        print(f"  {doc.metadata.get('source', 'unknown')} — {len(doc.page_content)} chars")

    return documents


def split_documents(documents: list, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Length: {len(chunk.page_content)} chars")
        print(chunk.page_content[:200])

    return chunks


def create_vector_store(chunks: list, persist_directory: str = PERSIST_DIR) -> Chroma:
    print("\nCreating embeddings and storing in ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store saved to '{persist_directory}'.")
    return vectorstore


def main():
    print("Starting ingestion pipeline...")
    documents = load_documents("docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)
    print("\nIngestion complete! Documents are ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()
