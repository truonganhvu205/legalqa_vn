from chunking import chunking
from FlagEmbedding import BGEM3FlagModel

def embeddings():
    chunks = chunking()
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
    sentences = [chunk.page_content for chunk in chunks]

    vectors = model.encode(
        sentences,
        batch_size=12,
        max_length=8192,
        )['dense_vecs']

    return chunks, vectors, model
