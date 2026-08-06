from embeddings import embeddings
import faiss

def vectorstore():
    vectors = embeddings().astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(1024)
    return index.add(vectors)
