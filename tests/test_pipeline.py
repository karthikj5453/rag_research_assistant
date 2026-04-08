import pytest
import os
import asyncio
from rag_pipeline import chunk_text

# ─── UNIT TESTS ─────────────────────────────────────────────────

def test_chunk_text_basic():
    """Verify that chunking works on simple page content."""
    dummy_content = [
        {"text": "This is a test document for chunking.", "page": 1}
    ]
    chunks = chunk_text(dummy_content, chunk_size=10, overlap=2)
    
    assert len(chunks) > 0
    assert chunks[0]["page"] == 1
    assert "text" in chunks[0]

def test_chunk_text_overlap():
    """Verify that overlap logic creates overlapping strings."""
    dummy_content = [
        {"text": "ABCDEFGHIJ", "page": 1}
    ]
    # Size 5, Overlap 2 -> Chunk 1: ABCDE, Chunk 2: DEFG...
    chunks = chunk_text(dummy_content, chunk_size=5, overlap=2)
    
    assert len(chunks) >= 2
    assert "DE" in chunks[0]["text"]
    assert chunks[1]["text"].startswith("DE")

# ─── ASYNC TESTS ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_environment():
    """Verify asyncio loop is functional."""
    val = await asyncio.sleep(0.1, result="ok")
    assert val == "ok"

# Note: In a real production CI, you would mock the Gemini API and VectorDB
# to test 'ingest_pdf_async' and 'ask_async' without API keys.
