import streamlit as st
import os
import asyncio
import base64
import fitz  # PyMuPDF
from rag_pipeline import ingest_pdf_async, ask_async, clear_knowledge_base_async, logger, px_session

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant [FINAL]", 
    page_icon="🤖",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background: linear-gradient(45deg, #4CAF50, #2E7D32); color: white; border: none; font-weight: bold; }
    .source-chunk { background-color: #161b22; padding: 20px; border-radius: 12px; border-left: 6px solid #4CAF50; margin-bottom: 25px; border: 1px solid #30363d; }
    .tag { padding: 5px 12px; border-radius: 8px; font-size: 0.9em; font-weight: bold; margin-right: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .page-tag { background-color: #2e7d32; color: white; }
    .score-tag { background-color: #1976d2; color: white; }
    .rrf-tag { background-color: #c62828; color: white; }
    .trace-card { background-color: #21262d; padding: 15px; border-radius: 10px; border: 1px dashed #4CAF50; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ─── Utility ────────────────────────────────────────────────────
def run_async(coro):
    return asyncio.run(coro)

def get_pdf_page_image(pdf_path, page_num):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        return base64.b64encode(pix.tobytes("png")).decode()
    except: return None

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=70)
    st.title("Admin Console")
    
    # LEVEL 4: Observability Link
    st.markdown("""<div class="trace-card">
    <h4 style="margin:0; color:#4CAF50;">📡 Phoenix Tracing</h4>
    <p style="font-size:0.85em; color:#8b949e;">Live monitoring of LLM calls and retrieval loops.</p>
    </div>""", unsafe_allow_html=True)
    if px_session:
        st.link_button("Open Observability Dashboard", px_session.url)
    else:
        st.error("Observability Server Offline")
    
    st.divider()
    st.subheader("📁 Multi-Doc Ingestion")
    uploaded_files = st.file_uploader("Index PDF Library", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            path = os.path.join("data", f.name)
            if not os.path.exists(path):
                with open(path, "wb") as file: file.write(f.getbuffer())
                with st.spinner(f"Indexing {f.name}..."):
                    run_async(ingest_pdf_async(path))
        st.success("Library Synced!")

    st.divider()
    if st.button("🔴 Factory Reset"):
        with st.spinner("Wiping environment..."):
            run_async(clear_knowledge_base_async())
            if os.path.exists("data"):
                import shutil
                shutil.rmtree("data")
                os.makedirs("data")
        st.success("System Reset Complete")
        st.rerun()

# ─── Main Interface ──────────────────────────────────────────────
st.title("🔬 Advanced RAG Assistant")
st.markdown("""
<div style='background-color: #23863622; padding: 10px 20px; border-radius: 10px; margin-bottom: 20px;'>
    <strong>Engine Active:</strong> Level 5 Hybrid Search (BM25 + Vector) | <strong>Observability:</strong> Arize Phoenix Active
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Research Query")
    query = st.text_input("Ask across all papers:", placeholder="Compare the performance metrics across models...")
    
    if st.button("Launch Hybrid Synthesis") and query:
        with st.spinner("🚀 Performing Hybrid Retrieval & Query Expansion..."):
            answer, context_items = run_async(ask_async(query))
        
        st.subheader("📝 Professional Synthesis")
        st.markdown(f"> {answer}")
        
        st.divider()
        st.subheader("📚 Evidence & Citations")
        for i, item in enumerate(context_items):
            page = item['metadata']['page']
            source = item['metadata']['source']
            score = item['score']
            
            st.markdown(f"""
            <div class="source-chunk">
                <span style="color:#4CAF50; font-weight:bold;">📄 {source}</span><br>
                <div style="margin-top:8px;">
                    <span class="tag page-tag">PAGE {page}</span>
                    <span class="tag score-tag">RERANK: {score:.2f}</span>
                </div>
                <p style="margin-top:15px; color:#c9d1d9; font-style:italic;">{item['text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"👁️ Visual Verification (Page {page})"):
                img = get_pdf_page_image(os.path.join("data", source), page)
                if img: st.markdown(f'<img src="data:image/png;base64,{img}" style="width:100%; border-radius:8px;">', unsafe_allow_html=True)
                else: st.warning("Visual preview source missing (likely BM25 meta artifact)")

with col2:
    st.header("⚙️ Search Strategy")
    st.write("**Strategy:** Hybrid-RRF")
    st.markdown("""
    - **Step 1:** Gemini Multi-Query Expansion
    - **Step 2:** Vector Search (all-MiniLM)
    - **Step 3:** BM25 Keyword Search
    - **Step 4:** Reciprocal Rank Fusion (RRF)
    - **Step 5:** Cross-Encoder Reranking
    """)
    st.success("✅ Full Stack Professional RAG Active")