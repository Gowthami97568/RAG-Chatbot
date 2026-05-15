RAG Chatbot
Approach :
I created a small structured CV and converted the text into embeddings using a SentenceTransformer model. These embeddings were stored in FAISS.
When a user asks a question, the system finds the most relevant part of the CV and returns it as the answer.
