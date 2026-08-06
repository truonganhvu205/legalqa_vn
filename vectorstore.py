from embeddings import embeddings
import faiss
import numpy as np

def vectorstore():
    chunks, vectors, model = embeddings()
    vectors = np.array(vectors).astype('float32')
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(1024)
    index.add(vectors)

    return index, chunks, model
