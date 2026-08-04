from loader import loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import os
from dotenv import load_dotenv

def tien_xu_ly():
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("uit_ds_c_2026")
    docs = loader()

    # CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_documents(docs)

    # EMBEDDINGS
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )

    # VECTOR DB
    db = FAISS.from_documents(
        documents=texts,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # RETRIEVER
    retriever = db.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': 5,
            'score_threshold': 0.2,
        },
    )

    return retriever
