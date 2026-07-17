# Your first Python file!

# variables
name = "Nida"
project = "PDF Question Answering System"

# list of skills we'll learn
skills = ["Python", "LangChain", "OpenAI", "FAISS", "FastAPI"]

# function
def introduce():
    print(f"Builder: {name}")
    print(f"Project: {project}")
    print(f"Skills to learn:")
    for skill in skills:
        print(f"  → {skill}")

# dictionary
project_info = {
    "name": project,
    "days": 8,
    "difficulty": "medium",
    "goal": "Build an AI system that answers questions from PDFs"
}

# call the function
introduce()

# print project info
print("\nProject details:")
for key, value in project_info.items():
    print(f"  {key}: {value}")