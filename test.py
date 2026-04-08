import asyncio
from rag_pipeline import ingest_pdf_async, ask_async, logger
import os

async def main():
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    pdf_name = "data/paper.pdf.pdf"

    if os.path.exists(pdf_name):
        logger.info("Starting Level 2 Async Test...")
        await ingest_pdf_async(pdf_name)

        query = "What is the main contribution of this paper?"
        print(f"\nQUERY: {query}")
        answer, context_items = await ask_async(query)

        print("\nANSWER:", answer)
        print("\nSOURCE CHUNKS (RE-RANKED):")
        for i, item in enumerate(context_items):
            page = item['metadata']['page']
            score = item['score']
            print(f"\n[Chunk {i+1} | Page {page} | Score {score:.4f}]:", item['text'][:150], "...")
    else:
        print(f"⚠️ Please place a PDF file at {pdf_name} to run the test.")

if __name__ == "__main__":
    asyncio.run(main())