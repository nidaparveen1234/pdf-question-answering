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

# ============================================
# STEP 4: ASK AI
# ============================================

def ask_ai(relevant_chunks, question):
    """
    Takes relevant chunks from FAISS
    Sends to Qwen with the question
    Returns answer
    """
    # join chunks into one context
    context = "\n\n".join(relevant_chunks)
    
    # build prompt
    prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the context below.
If answer is not in context say 
'I could not find this in the document.'

Context:
{context}

Question: {question}

Answer:"""

    # send to local Qwen model
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return response['message']['content']

# ============================================
# STEP 5: MAIN — CONNECT EVERYTHING
# ============================================

def main():
    print("🤖 PDF Question Answering System")
    print("=" * 50)
    
    # step 1: process the PDF
    chunks = process_pdf("test.pdf")
    
    # step 2: create embeddings and store in FAISS
    index = create_vector_store(chunks)
    
    # step 3: question loop
    # user can keep asking questions until they type 'exit'
    print("\n✅ System ready! Ask questions about your PDF.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # get question from user
        question = input("❓ Your question: ")
        
        # exit condition
        if question.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        # skip empty questions
        if question.strip() == "":
            continue
        
        print("🔍 Searching PDF...")
        
        # search FAISS for relevant chunks
        relevant_chunks = search_relevant_chunks(
            index, chunks, question
        )
        
        print("🤖 Asking AI...")
        
        # ask AI with those chunks
        answer = ask_ai(relevant_chunks, question)
        
        print(f"\n💬 Answer: {answer}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()