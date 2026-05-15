cv_data = [
    "Name: Gowthami",
    "Education: B.Tech in Artificial Intelligence & Data Science from Seshadri Rao GUdlavalleru Engineering College",
    "Skills: Python, Machine Learning, Deep Learning, SQL",
    "Projects: CNN image classifier, RAG chatbot system",
    "Experience: Internship at Vinti Web Solutions as a Frontend Developer"
]
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(cv_data)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
query = input("Ask a question: ")
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=1)
results = [cv_data[i] for i in I[0]]
answer = results[0]
print("Answer:", answer)