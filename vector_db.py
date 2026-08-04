from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from chunking import chunking
from embeddings import embeddings
import os

INDEX_PATH = "faiss_index"

def vector_db():
    emb = embeddings()
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

    db = FAISS.from_documents(
        documents=chunking(),
        embedding=emb,
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    db.save_local(INDEX_PATH)
    return db
