from vectorstore import vectorstore
import faiss
import numpy as np

def retriever():
    index, chunks, model = vectorstore()

    def retrieve(question, k=5, score_threshold=0.2):
        query = model.encode(
            [question],
            max_length=512,
            )['dense_vecs']
        query = np.array(query).astype('float32')
        faiss.normalize_L2(query)

        scores, indices = index.search(query, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < score_threshold:
                continue
            results.append(chunks[idx])
        return results

    def retrieve_batch(questions, k=5, score_threshold=0.2, batch_size=32):
        queries = model.encode(
            questions,
            batch_size=batch_size,
            max_length=512,
            )['dense_vecs']
        queries = np.array(queries).astype('float32')
        faiss.normalize_L2(queries)

        scores, indices = index.search(queries, k)
        all_results = []

        for i in range(len(questions)):
            results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx == -1 or score < score_threshold:
                    continue
                results.append(chunks[idx])
            all_results.append(results)
        return all_results

    return retrieve, retrieve_batch
