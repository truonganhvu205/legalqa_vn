from embeddings import embeddings
import faiss

vectors = embeddings().astype("float32")
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(1024)
index.add(vectors)
