import asyncio
import os
from rag_pipeline import ingest_pdf_async, ask_async, px_session

async def verify():
    print("💎 FINAL LEVEL 5 VERIFICATION")
    if px_session:
        print(f"Observability URL: {px_session.url}")
    else:
        print("Observability URL: Offline")
    
    pdf_path = "data/paper.pdf.pdf"
    if os.path.exists(pdf_path):
        print("Indexing document...")
        await ingest_pdf_async(pdf_path)
        
        print("\nTesting Hybrid Search + Expansion...")
        query = "What is the implementation strategy?"
        answer, context = await ask_async(query)
        
        print(f"\nAnswer: {answer[:150]}...")
        print(f"\nSources Found: {len(context)}")
        for i, c in enumerate(context):
            print(f"- {c['metadata']['source']} (P{c['metadata']['page']}) | Rerank Score: {c['score']:.2f}")
            
        print("\n✅ LEVEL 5 FUNCTIONAL")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    asyncio.run(verify())
