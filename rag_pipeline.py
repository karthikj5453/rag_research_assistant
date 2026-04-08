import os
import fitz  # PyMuPDF
import chromadb
import logging
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv

# ─── LEVEL 4: OBSERVABILITY (PHOENIX) ───────────────────────────
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace as otel_trace

# Setup Phoenix and Instrumentation
print("🚀 Initializing Arize Phoenix Observability...")
try:
    px_session = px.launch_app()
    if px_session is None:
        px_session = px.active_session()
    
    # Standardised Phoenix OTEL Registration
    tracer_provider = register(
        project_name="rag-research-assistant",
        endpoint="http://localhost:6006/v1/traces"
    )
    otel_trace.set_tracer_provider(tracer_provider)
except Exception as e:
    print(f"⚠️ Phoenix Bridge issue: {e}")
    px_session = None

# Instrument Gemini SDK
GoogleGenAIInstrumentor().instrument()
tracer = otel_trace.get_tracer(__name__)

# ─── LOGGING SETUP ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Pipeline")

# ─── INITIALIZATION ──────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logger.info("Loading models (Professional Stack Active)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

bm25_index = None
bm25_docs = []


# ─── STEP 1: Ingestion with BM25 Indexing ───────────────────────
async def extract_text_from_pdf_async(pdf_path):
    def _extract():
        doc = fitz.open(pdf_path)
        content = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text: content.append({"text": text, "page": i + 1})
        return content
    return await asyncio.to_thread(_extract)

def chunk_text(pages_content, chunk_size=1000, overlap=200):
    chunks = []
    for page in pages_content:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "page": page["page"]})
            if end >= len(text): break
            start += chunk_size - overlap
    return chunks

async def ingest_pdf_async(pdf_path):
    global bm25_index, bm25_docs
    with tracer.start_as_current_span("Ingest_PDF") as span:
        pages_content = await extract_text_from_pdf_async(pdf_path)
        
        # Cleanup existing chunks for this specific source
        def _cleanup():
            existing = collection.get(where={"source": os.path.basename(pdf_path)})
            if existing['ids']:
                collection.delete(ids=existing['ids'])
        await asyncio.to_thread(_cleanup)

        chunks = chunk_text(pages_content)
        texts = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"], "source": os.path.basename(pdf_path)} for c in chunks]
        
        embeddings = await asyncio.to_thread(lambda: embedder.encode(texts, show_progress_bar=False).tolist())
        
        def _store():
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=[f"{os.path.basename(pdf_path)}_p{c['page']}_c{i}" for i, c in enumerate(chunks)]
            )
        await asyncio.to_thread(_store)
        
        # Standardised BM25 Refresh
        bm25_docs.extend(chunks)
        tokenized_corpus = [doc['text'].lower().split() for doc in bm25_docs]
        bm25_index = BM25Okapi(tokenized_corpus)
        return len(chunks)


# ─── STEP 2: Query Expansion ────────────────────────────────────
async def expand_query_async(query):
    with tracer.start_as_current_span("Query_Expansion") as span:
        prompt = f"Generate 3 diverse alternative search queries for: '{query}'. Provide only queries, one per line."
        try:
            response = await gemini_model.generate_content_async(prompt)
            expanded = [query] + [q.strip() for q in response.text.split('\n') if q.strip()]
            return expanded[:4]
        except:
            return [query]


# ─── STEP 3: Hybrid Retrieval (RRF) ─────────────────────────────
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    def add_score(ranked_list):
        for rank, item in enumerate(ranked_list):
            doc_id = f"{item['metadata']['source']}_p{item['metadata']['page']}"
            key = (item['text'], doc_id)
            scores[key] = scores.get(key, 0) + (1.0 / (k + rank))
    add_score(vector_results)
    add_score(bm25_results)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for (text, doc_id), score in sorted_scores:
        source, page = doc_id.rsplit('_p', 1)
        fused.append({"text": text, "metadata": {"source": source, "page": int(page)}, "rrf_score": score})
    return fused

async def retrieve_hybrid_async(query, top_k_final=5):
    with tracer.start_as_current_span("Hybrid_Retrieval") as span:
        expanded_queries = await expand_query_async(query)
        vector_candidates = []
        for q in expanded_queries:
            emb = await asyncio.to_thread(lambda: embedder.encode([q]).tolist())
            res = await asyncio.to_thread(lambda: collection.query(query_embeddings=emb, n_results=10))
            for i in range(len(res['documents'][0])):
                vector_candidates.append({"text": res['documents'][0][i], "metadata": res['metadatas'][0][i]})

        bm25_candidates = []
        if bm25_index:
            top_bm25 = bm25_index.get_top_n(query.lower().split(), bm25_docs, n=15)
            for doc in top_bm25:
                bm25_candidates.append({"text": doc['text'], "metadata": {"source": "BM25_Index", "page": doc['page']}})

        fused = reciprocal_rank_fusion(vector_candidates, bm25_candidates)
        pairs = [[query, f["text"]] for f in fused[:15]]
        scores = await asyncio.to_thread(lambda: reranker.predict(pairs))
        for i in range(len(scores)): fused[i]["score"] = float(scores[i])
        fused.sort(key=lambda x: x.get("score", 0), reverse=True)
        return fused[:top_k_final]


# ─── FINAL PIPELINE ─────────────────────────────────────────────
async def ask_async(query):
    with tracer.start_as_current_span("RAG_Engine_Ask") as span:
        context_items = await retrieve_hybrid_async(query)
        if not context_items: return "No relevant documents found.", []
        
        ctx_str = ""
        for item in context_items:
            ctx_str += f"\nSource: {item['metadata']['source']} (P{item['metadata']['page']})\n{item['text']}\n"
            
        prompt = f"Using this context:\n{ctx_str}\nAnswer: {query}"
        response = await gemini_model.generate_content_async(prompt)
        return response.text, context_items

async def clear_knowledge_base_async():
    global bm25_index, bm25_docs
    def _clear():
        global bm25_index, bm25_docs
        res = collection.get()
        if res['ids']: collection.delete(ids=res['ids'])
        bm25_index = None
        bm25_docs = []
    await asyncio.to_thread(_clear)
    logger.info("Knowledge base cleared.")