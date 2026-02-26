from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(faqs):
    embeddings = model.encode(faqs)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings

def search(query, index, faqs):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return faqs[I[0][0]]