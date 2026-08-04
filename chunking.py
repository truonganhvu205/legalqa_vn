from loader import loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking():
    docs = loader()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    texts = text_splitter.split_documents(docs)
    return texts
