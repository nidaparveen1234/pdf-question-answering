from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# ============================================
# STEP 1: READ AND CHUNK PDF
# ============================================

def process_pdf(file_path):
    """
    Reads the PDF and splits into chunks.
    We built this on Day 2 — same code!
    """
    # read PDF
    print(f"\n📄 Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    
    raw_text = ""
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    
    print(f"✅ Read {len(reader.pages)} pages")
    print(f"✅ Total characters: {len(raw_text)}")
    
    # split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = splitter.split_text(raw_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    return chunks