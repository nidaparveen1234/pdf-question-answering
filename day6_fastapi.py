from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import io

# ============================================
# SETUP
# ============================================

# create FastAPI app — same as express() in Node
app = FastAPI()

# allow React frontend to talk to this backend
# same as cors() in Express
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load embedding model once when server starts
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# global storage — holds chunks and FAISS index
# in a real app we'd use a database
# for now this lives in memory while server runs
chunks_store = []
index_store = None

# ============================================
# MODELS — like defining req.body shape
# ============================================

# this defines what the question request body looks like
class QuestionRequest(BaseModel):
    question: str

# ============================================
# HELPER FUNCTIONS — same as Day 5
# ============================================

def extract_text_from_pdf(file_bytes):
    """
    Takes PDF as bytes (uploaded file)
    Extracts all text from it
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def split_into_chunks(text):
    """
    Splits text into chunks
    Same as Day 2
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)

def create_faiss_index(chunks):
    """
    Creates embeddings and stores in FAISS
    Same as Day 4
    """
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_chunks(index, chunks, question, top_k=3):
    """
    Searches FAISS for relevant chunks
    Same as Day 4
    """
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    D, I = index.search(question_embedding, top_k)
    relevant = []
    for i in I[0]:
        if i < len(chunks):
            relevant.append(chunks[i])
    return relevant

def ask_qwen(relevant_chunks, question):
    """
    Sends relevant chunks + question to Qwen
    Same as Day 5
    """
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context say
'I could not find this in the document.'

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# ============================================
# ROUTES — same idea as Express routes
# ============================================

# test route — like app.get('/', ...) in Express
@app.get("/")
def home():
    return {"message": "PDF QA System running!"}

# upload PDF route
# user uploads PDF → we process it → store in memory
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives PDF file from frontend
    Reads it, chunks it, creates FAISS index
    Stores everything in global variables
    """
    global chunks_store, index_store

    # check file is actually a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    try:
        # read the uploaded file as bytes
        file_bytes = await file.read()

        # extract text
        print(f"📄 Processing: {file.filename}")
        raw_text = extract_text_from_pdf(file_bytes)

        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )

        # split into chunks
        chunks_store = split_into_chunks(raw_text)
        print(f"✅ Created {len(chunks_store)} chunks")

        # create FAISS index
        index_store = create_faiss_index(chunks_store)
        print(f"✅ FAISS index created")

        return {
            "message": "PDF uploaded and processed successfully!",
            "chunks": len(chunks_store),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ask question route
# user sends question → we search FAISS → ask Qwen → return answer
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Receives question from frontend
    Searches FAISS for relevant chunks
    Asks Qwen AI
    Returns answer
    """
    global chunks_store, index_store

    # check PDF has been uploaded first
    if index_store is None or len(chunks_store) == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first"
        )

    try:
        print(f"❓ Question: {request.question}")

        # search FAISS
        relevant_chunks = search_chunks(
            index_store,
            chunks_store,
            request.question
        )

        # ask Qwen
        answer = ask_qwen(relevant_chunks, request.question)
        print(f"💬 Answer: {answer[:100]}...")

        return {
            "question": request.question,
            "answer": answer,
            "chunks_used": len(relevant_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Loading embedding model...
#Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
#Loading weights: 100%|██████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3407.64it/s]
#✅ Embedding model loaded!
#INFO:     Started server process [16340]
#INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
# INFO:     127.0.0.1:54258 - "GET /docs HTTP/1.1" 200 OK
# INFO:     127.0.0.1:53297 - "GET /openapi.json HTTP/1.1" 200 OK
# INFO:     127.0.0.1:59188 - "GET /docs HTTP/1.1" 200 OK
# INFO:     127.0.0.1:59188 - "GET /openapi.json HTTP/1.1" 200 OK
# INFO:     127.0.0.1:54585 - "GET / HTTP/1.1" 200 OK
# INFO:     127.0.0.1:65438 - "POST /ask HTTP/1.1" 400 Bad Request
# 📄 Processing: test.pdf
# ✅ Created 34 chunks
# ✅ FAISS index created
# INFO:     127.0.0.1:57679 - "POST /upload HTTP/1.1" 200 OK
# ❓ Question: string
# 💬 Answer: I could not find this in the document....
# INFO:     127.0.0.1:58996 - "POST /ask HTTP/1.1" 200 OK
# ❓ Question: what is ml
# 💬 Answer: Machine Learning (ML) is an interdisciplinary field that focuses on the study of algorithms and comp...
# INFO:     127.0.0.1:52126 - "POST /ask HTTP/1.1" 200 OK
