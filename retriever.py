from vector_db import vector_db

def retriever():
    return vector_db().as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': 5,
            'score_threshold': 0.2,
        },
    )
