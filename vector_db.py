from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from chunking import chunking
from embeddings import embeddings

def vector_db():
    db = FAISS.from_documents(
        documents=chunking(),
        embedding=embeddings(),
        distance_strategy=DistanceStrategy.COSINE,
    )

    return db
