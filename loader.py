import json
from langchain_core.documents import Document

def loader():
    with open("./datasets/warmup.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for id, item in data.items():
        docs.append(
            Document(
                page_content=f"{item['answer']}",
                metadata={},
                id=id,
            )
        )

    return docs
