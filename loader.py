import json
from langchain_core.documents import Document

def loader():
    with open("./datasets/public_test/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for id, item in data.items():
        docs.append(
            Document(
                page_content=(
                    f"question: {item['question']}\n"
                    f"answer: {item['answer']}"
                ),
                metadata={
                    'question_id': id,
                },
                id=id,
            )
        )

    return docs

if __name__ == '__main__':
    print(loader())
