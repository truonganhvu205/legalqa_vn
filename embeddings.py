from chunking import chunking
from FlagEmbedding import BGEM3FlagModel

def embeddings():
    docs = chunking()
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
    sentences = [doc.page_content for doc in docs]

    return model.encode(sentences,
                        batch_size=12,
                        max_length=8192,
                        )['dense_vecs']

if __name__ == '__main__':
    print(embeddings())
