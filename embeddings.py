# from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# import os
# from dotenv import load_dotenv

def embeddings():
    # load_dotenv()

    # return HuggingFaceEndpointEmbeddings(
    #     model="BAAI/bge-m3",
    #     huggingfacehub_api_token=os.getenv("uit_ds_c_2026"),
    # )

    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
