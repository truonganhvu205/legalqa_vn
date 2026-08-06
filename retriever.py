from vectorstore import vectorstore

def retriever():
    vectors = vectorstore()
    return vectors.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': 5,
            'score_threshold': 0.2,
        },
    )
