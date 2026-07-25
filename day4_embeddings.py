import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# load the embedding model
# this downloads a small model first time, then works offline
print("loading embedding model....")
model = SentenceTransformer('all-MiniLM-L6-v2')#a small free model that converts text to 384 numbers
print("Model loaded!")


# step 2 create embedding

def create_embeddings(chunks):
   """
    Takes a list of text chunks
    Converts each one into a list of numbers
    Returns all embeddings as a numpy array
    """
   print(f"Creating embeddings for {len(chunks)} chunks...")
   embeddings = model.encode(chunks)#takes your list of text chunks
   print(f" Each chunk converted to {embeddings.shape[1]} numbers")
   return embeddings

# step 3 store in Faiss

# ============================================
# STEP 3: STORE IN FAISS
# ============================================

def store_in_faiss(embeddings):
    """
    Takes all the embeddings (numbers)
    Stores them in FAISS so we can search later
    
    Think of FAISS like a library index:
    - Each embedding = one book's summary
    - FAISS organizes them so we can find
      similar ones super fast
    """
    # embeddings.shape[1] = 384
    # we tell FAISS each embedding has 384 numbers
    dimension = embeddings.shape[1]
    
    # create a FAISS index
    # IndexFlatL2 = finds similar vectors using distance calculation
    index = faiss.IndexFlatL2(dimension)
    
    # add all embeddings to the index
    index.add(embeddings)
    
    print(f"✅ Stored {index.ntotal} chunks in FAISS")
    return index
#dimension = embeddings.shape[1] → gets 384 (number of dimensions)
#faiss.IndexFlatL2 → creates the search index. L2 means it measures 
# distance between vectors — closer distance = more similar meaning
#index.add(embeddings) → loads all our chunk embeddings into FAISS
#index.ntotal → how many chunks are stored

# ============================================
# STEP 4: SEARCH FAISS
# ============================================

def search_faiss(index, chunks, question, top_k=3):
    """
    Takes a question
    Converts it to embedding
    Searches FAISS for most similar chunks
    Returns top 3 most relevant chunks
    
    top_k = how many chunks to return
    """
    # convert question to embedding
    # same process as converting PDF chunks
    question_embedding = model.encode([question])
    
    # search FAISS
    # D = distances (how similar)
    # I = indices (which chunks matched)
    D, I = index.search(question_embedding, top_k)
    
    # get the actual text chunks using the indices
    relevant_chunks = []
    for i in I[0]:
        if i < len(chunks):
            relevant_chunks.append(chunks[i])
    
    print(f"✅ Found {len(relevant_chunks)} relevant chunks")
    return relevant_chunks
#model.encode([question]) → converts question to 384 numbers — same way we converted chunks
#index.search(question_embedding, top_k) → searches FAISS for 3 most similar chunks
#D → distances. Smaller = more similar
#I → indices. Which position in our chunks list matched
#I[0] → first row of results (we only searched one question)
#We loop through indices and grab the actual text chunks

# ============================================
# STEP 5: ASK AI WITH CONTEXT
# ============================================

def ask_ai(relevant_chunks, question):
    """
    Takes the relevant chunks from FAISS
    Joins them into one context block
    Sends to our local Qwen model
    Returns the answer
    """
    # join all relevant chunks into one big context
    context = "\n\n".join(relevant_chunks)
    
    # build the prompt
    prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context, say 
'I could not find this in the document.'

Context:
{context}

Question: {question}

Answer:"""

    # send to our local ollama model
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
#"\n\n".join(relevant_chunks) → combines 3 chunks into one text block with empty lines between them
#prompt → packages context + question together into one message
#ollama.chat() → sends to our local Qwen model
#response['message']['content'] → extracts just the text answer

# ============================================
# STEP 6: CONNECT EVERYTHING
# ============================================

def main():
    print("🤖 PDF Question Answering System\n")
    print("=" * 50)# this line is for decoration
    
    # some fake PDF chunks for testing
    # tomorrow we connect real PDF from day 2
    chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Supervised learning uses labeled data to train models. Examples include classification and regression problems.",
        "Unsupervised learning finds hidden patterns in unlabeled data. Examples include clustering and dimensionality reduction.",
        "Reinforcement learning trains models through rewards and penalties. It is used in game playing and robotics.",
        "Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data.",
        "Natural language processing is a branch of AI that helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from images and videos.",
        "FAISS is a library developed by Facebook for efficient similarity search of dense vectors.",
        "Embeddings convert text into numbers that represent the meaning of that text in a high dimensional space.",
        "RAG stands for Retrieval Augmented Generation. It combines search with AI to answer questions from documents."
    ]
    
    # step 1: create embeddings for all chunks
    print("\nStep 1: Creating embeddings...")
    embeddings = create_embeddings(chunks)
    
    # step 2: store in FAISS
    print("\nStep 2: Storing in FAISS...")
    index = store_in_faiss(embeddings)
    
    # step 3: ask questions
    print("\nStep 3: Ready to answer questions!")
    print("=" * 50)
    
    # test questions
    questions = [
        "What is machine learning?",
        "What is reinforcement learning?",
        "What is RAG?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 30)
        
        # search FAISS for relevant chunks
        relevant_chunks = search_faiss(index, chunks, question)
        
        # ask AI with those chunks
        answer = ask_ai(relevant_chunks, question)
        
        print(f"💬 Answer: {answer}")
        print("-" * 30)# this line is for decoration
    
    print("\n🎉 Day 4 complete! FAISS + Embeddings working!")

if __name__ == "__main__":
    main()