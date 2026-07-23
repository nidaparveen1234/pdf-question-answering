import ollama

# the model we downloaded
MODEL = "qwen2.5:3b"

# ============================================
# STEP 1: SIMPLE QUESTION
# ============================================

def ask_ai(question):
    """
    Sends a question to our local AI and gets an answer.
    No internet. No API key. Runs on our laptop.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep answers short and clear."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    # extract just the text answer
    return response['message']['content']

# ============================================
# STEP 2: QUESTION WITH PDF CONTEXT
# ============================================

def ask_ai_with_context(context, question):
    """
    This is the core of our PDF QA system.
    We give AI a chunk from the PDF + a question.
    AI answers based ONLY on that chunk.
    
    This is RAG in its simplest form:
    - Retrieve the relevant chunk
    - Give it to AI as context
    - AI Generates answer from that context
    """
    
    prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say 'I could not find this in the document.'
Keep your answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""
    
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return response['message']['content']

# ============================================
# STEP 3: TEST EVERYTHING
# ============================================

def main():
    print("🤖 Testing local AI with Qwen2.5:3b\n")
    print("=" * 50)
    
    # Test 1: simple question
    print("\nTest 1: Simple question")
    print("-" * 30)
    question = "What is artificial intelligence in one sentence?"
    print(f"Question: {question}")
    answer = ask_ai(question)
    print(f"Answer: {answer}")
    
    # Test 2: question with context — RAG preview
    print("\nTest 2: Question with PDF context")
    print("-" * 30)
    
    # pretend this is a chunk from our PDF
    fake_pdf_chunk = """
    Machine learning is a subset of artificial intelligence 
    that enables computers to learn patterns from data without 
    being explicitly programmed. There are three main types:
    supervised learning where the model learns from labeled data,
    unsupervised learning where the model finds patterns in 
    unlabeled data, and reinforcement learning where the model 
    learns by trial and error through rewards and penalties.
    """
    
    question = "What are the three types of machine learning?"
    print(f"Context: {fake_pdf_chunk}")
    print(f"Question: {question}")
    answer = ask_ai_with_context(fake_pdf_chunk, question)
    print(f"Answer: {answer}")
    
    print("\n" + "=" * 50)
    print("🎉 Local AI working! Ready for Day 4.")

if __name__ == "__main__":
    main()