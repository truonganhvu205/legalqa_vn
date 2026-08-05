# submit.py
import json
from retriever import retriever as get_retriever
from llm import load_client, generate

def load_questions(path="./datasets/warmup.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_submission(input_path="./datasets/warmup.json", output_path="submission.json"):
    questions = load_questions(input_path)
    retriever = get_retriever()
    client = load_client()

    submission = {}

    for qid, item in questions.items():
        question = item["question"].strip().lower()

        context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
        answer = generate(client, question, context)

        submission[qid] = {"answer": answer}
        print(f"Đã xử lý {qid}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất. Đã lưu {len(submission)} câu trả lời vào {output_path}")

if __name__ == "__main__":
    make_submission()
