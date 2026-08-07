from embeddings import embeddings
import faiss
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import os
import pickle

INDEX_PATH = 'faiss_index.bin'
CHUNKS_PATH = 'chunks.pkl'

def vectorstore():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)

        model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
        return index, chunks, model

    chunks, vectors, model = embeddings()
    vectors = np.array(vectors).astype('float32')
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(1024)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

    return index, chunks, model
