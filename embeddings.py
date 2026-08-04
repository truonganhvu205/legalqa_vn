from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

def embeddings():
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("uit_ds_c_2026")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )
    
    return embeddings
