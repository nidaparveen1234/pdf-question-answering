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

# ============================================
# STEP 2: CREATE EMBEDDINGS AND STORE IN FAISS
# ============================================

def create_vector_store(chunks):
    """
    Converts chunks to embeddings
    Stores in FAISS
    We built this on Day 4 — same code!
    """
    # create embeddings
    print("\nCreating embeddings...")
    embeddings = model.encode(chunks)
    print(f"✅ Created embeddings: {embeddings.shape}")
    
    # store in FAISS
    print("Storing in FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ Stored {index.ntotal} chunks in FAISS")
    
    return index

# ============================================
# STEP 3: SEARCH FAISS
# ============================================

def search_relevant_chunks(index, chunks, question, top_k=3):
    """
    Converts question to embedding
    Searches FAISS for most similar chunks
    Returns top 3 relevant chunks
    """
    # convert question to embedding
    question_embedding = model.encode([question])
    
    # search FAISS
    D, I = index.search(question_embedding, top_k)
    
    # get actual text chunks
    relevant_chunks = []
    for i in I[0]:
        if i < len(chunks):
            relevant_chunks.append(chunks[i])
    
    return relevant_chunks