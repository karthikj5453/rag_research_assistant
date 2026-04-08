import asyncio
import os
import base64
from rag_pipeline import ingest_pdf_async, ask_async, logger

async def verify():
    print("🚀 LEVEL 3 FUNCTIONAL VERIFICATION")
    
    # 1. Check if we can ingest a second document (if possible, or just re-ingest)
    pdf_path = "data/paper.pdf.pdf"
    if os.path.exists(pdf_path):
        print(f"--- Testing Ingestion ---")
        chunks = await ingest_pdf_async(pdf_path)
        print(f"SUCCESS: Ingested {chunks} chunks.")
    
    # 2. Test Multi-Doc / Synthesis Logic
    print(f"\n--- Testing Synthesis ---")
    query = "Summarize the findings."
    answer, context = await ask_async(query)
    if answer and len(context) > 0:
        print(f"SUCCESS: Synthesized answer received with {len(context)} sources.")
        print(f"Sample Answer: {answer[:100]}...")
    else:
        print("FAILED: No answer or context retrieved.")

    # 3. Test PDF Rendering (The visual preview feature)
    print(f"\n--- Testing Visual Rendering ---")
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_b64 = base64.b64encode(pix.tobytes()).decode()
        if len(img_b64) > 1000:
            print("SUCCESS: PDF page rendered to base64 image.")
        else:
            print("FAILED: Image encoding too small.")
    except Exception as e:
        print(f"FAILED: Rendering error: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    asyncio.run(verify())
