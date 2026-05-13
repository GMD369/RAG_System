import os
import io
import contextlib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "rag-system-secret-key")

# ── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ingestion")
def ingestion():
    docs = []
    docs_path = "docs"
    if os.path.exists(docs_path):
        docs = [f for f in os.listdir(docs_path) if f.endswith((".txt", ".pdf"))]
    return render_template("ingestion.html", docs=docs)

@app.route("/retrieval")
def retrieval():
    return render_template("retrieval.html")

@app.route("/answer")
def answer_page():
    return render_template("answer.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/chunking")
def chunking():
    return render_template("chunking.html")

# ── API: Ingestion ────────────────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            from ingestion_pipeline import load_documents, split_documents, create_vector_store
            docs = load_documents("docs")
            chunks = split_documents(docs)
            create_vector_store(chunks)

        log = output.getvalue()
        doc_count = len(docs)
        chunk_count = len(chunks)
        return jsonify({
            "success": True,
            "log": log,
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "steps": [
                {"label": "Load Documents", "detail": f"{doc_count} file(s) loaded"},
                {"label": "Split into Chunks", "detail": f"{chunk_count} chunks created (size=800, overlap=150)"},
                {"label": "Generate Embeddings", "detail": "all-MiniLM-L6-v2 (384 dimensions)"},
                {"label": "Store in ChromaDB", "detail": f"{chunk_count} vectors persisted to db/chroma_db"},
            ],
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc), "log": output.getvalue()})

# ── API: Retrieval ────────────────────────────────────────────────────────────

@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    query = (request.json or {}).get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "No query provided"})
    try:
        from utils import get_vectorstore
        score_threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        db = get_vectorstore()
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        docs = retriever.invoke(query)
        results = [
            {
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "content": d.page_content,
                "length": len(d.page_content),
            }
            for d in docs
        ]
        return jsonify({"success": True, "results": results, "count": len(results), "query": query})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})

# ── API: Answer ───────────────────────────────────────────────────────────────

@app.route("/api/answer", methods=["POST"])
def api_answer():
    query = (request.json or {}).get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "No query provided"})
    try:
        from retrieval_pipeline import retrieve, answer
        docs = retrieve(query)
        result = answer(query, docs)
        context = [
            {
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "content": d.page_content,
            }
            for d in docs
        ]
        return jsonify({"success": True, "answer": result, "context": context, "doc_count": len(docs)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})

# ── API: History-Aware Chat ───────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    history_data = data.get("history", [])
    if not message:
        return jsonify({"success": False, "error": "No message provided"})
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from history_aware_generation import rewrite_query, retrieve, generate_answer
        from utils import get_llm

        chat_history = []
        for item in history_data:
            if item["role"] == "user":
                chat_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                chat_history.append(AIMessage(content=item["content"]))

        model = get_llm()
        search_question = rewrite_query(model, message, chat_history)
        docs = retrieve(search_question)
        reply = generate_answer(model, message, docs, chat_history)

        context = [
            {
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "content": d.page_content,
            }
            for d in docs
        ]
        return jsonify({
            "success": True,
            "reply": reply,
            "search_question": search_question,
            "rewritten": search_question != message,
            "context": context,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})

# ── API: Chunking ─────────────────────────────────────────────────────────────

@app.route("/api/chunk", methods=["POST"])
def api_chunk():
    data = request.json or {}
    text = data.get("text", "").strip()
    method = data.get("method", "recursive")
    if not text:
        return jsonify({"success": False, "error": "No text provided"})
    try:
        chunks = []

        if method == "character":
            from langchain_text_splitters import CharacterTextSplitter
            splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=0)
            chunks = splitter.split_text(text)

        elif method == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=200,
                chunk_overlap=30,
            )
            chunks = splitter.split_text(text)

        elif method == "semantic":
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from utils import get_embeddings
                splitter = SemanticChunker(
                    embeddings=get_embeddings(),
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=70,
                )
                chunks = splitter.split_text(text)
            except ImportError:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
                chunks = splitter.split_text(text)

        elif method == "agentic":
            from utils import get_llm
            llm = get_llm()
            prompt = (
                "You are a text chunking expert. Split this text into logical chunks.\n"
                "Rules:\n- Each chunk ~200 chars or less\n- Split at natural topic boundaries\n"
                "- Put <<<SPLIT>>> between chunks\n\nText:\n" + text +
                "\n\nReturn text with <<<SPLIT>>> markers:"
            )
            response = llm.invoke(prompt)
            chunks = [c.strip() for c in response.content.split("<<<SPLIT>>>") if c.strip()]

        return jsonify({
            "success": True,
            "chunks": [{"content": c, "length": len(c)} for c in chunks],
            "count": len(chunks),
            "method": method,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
