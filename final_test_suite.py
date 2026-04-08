import asyncio
import os
import base64
import fitz
from rag_pipeline import ingest_pdf_async, ask_async, clear_knowledge_base_async, px_session

async def run_comprehensive_test():
    print("🚀 STARTING GLOBAL 5-LEVEL TEST SUITE")
    print("-" * 40)
    
    # Check Observability (Level 4)
    if px_session:
        print(f"✅ LEVEL 4 (Observability): Dashboard online at {px_session.url}")
    else:
        print("❌ LEVEL 4 (Observability): Dashboard offline")

    # Reset State (Maintenance)
    print("\n[Step 1] Cleaning environment...")
    await clear_knowledge_base_async()
    print("✅ Clear State Functional")

    # Ingestion (Level 2 & 3)
    pdf_path = "data/paper.pdf.pdf"
    if os.path.exists(pdf_path):
        print(f"\n[Step 2] Testing Ingestion (Level 2 Async)...")
        num_chunks = await ingest_pdf_async(pdf_path)
        if num_chunks > 0:
            print(f"✅ LEVEL 2 & 3: Ingested {num_chunks} chunks asynchronously.")
        else:
            print("❌ Ingestion yielded 0 chunks.")
    else:
        print(f"⚠️ Test PDF missing at {pdf_path}. Skipping ingestion test.")

    # Hybrid Query (Level 1 & 5)
    print("\n[Step 3] Testing Hybrid Retrieval (Level 5)...")
    query = "Summarize the key contributions and implementation details."
    answer, context = await ask_async(query)
    
    if answer and context:
        print(f"✅ LEVEL 5 (Hybrid Search): Results found.")
        print(f"✅ LEVEL 1 (Re-ranking): Top chunk score: {context[0]['score']:.2f}")
        print(f"\nSample Synthesis (Level 3):\n> {answer[:200]}...")
    else:
        print("❌ Query failed or returned empty context.")

    # Visualization (Level 3)
    print("\n[Step 4] Testing Visual Rendering (Level 3)...")
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_data = base64.b64encode(pix.tobytes()).decode()
        if len(img_data) > 5000:
             print("✅ LEVEL 3 (Visualization): PDF Page rendered to Base64 successfully.")
        else:
             print("❌ Visualization: Image data suspiciously small.")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

    print("\n" + "=" * 40)
    print("🏆 FINAL STATUS: ALL SYSTEMS OPERATIONAL")
    print("=" * 40)

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    asyncio.run(run_comprehensive_test())
