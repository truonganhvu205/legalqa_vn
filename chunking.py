from loader import loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking():
    docs = loader()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    return text_splitter.split_documents(docs)

if __name__ == '__main__':
    print(chunking()[0].page_content)
