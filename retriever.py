from vectorstore import vectorstore
import faiss
import numpy as np

def retriever():
    index, chunks, model = vectorstore()

    def retrieve(question, k=5, score_threshold=0.2):
        query = model.encode(
            [question],
            max_length=8192,
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
    return retrieve

if __name__ == '__main__':
    r = retriever()
    docs = r('Phó chủ tịch công đoàn có được quyền ký thỏa ước lao động tập thể không?')
    print(len(docs))
