from llm import llm
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("uit_ds_c_2026")

def rag_pipeline():
    print(llm())

if __name__ == "__main__":
    rag_pipeline()
