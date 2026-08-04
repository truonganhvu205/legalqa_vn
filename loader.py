import json
from langchain_core.documents import Document

with open("./datasets/warmup.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []

for id, item in data.items():
    docs.append(
        Document(
            page_content=f"answer: {item['answer']}",
            metadata={"question_id": id}
        )
    )

print(docs[0].page_content)
print(docs[0].metadata)
