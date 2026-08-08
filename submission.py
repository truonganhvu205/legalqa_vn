import json
import os
from retriever import retriever as get_retriever
from llm import load_model, generate

def load_questions(path="./datasets/public_test/train.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_submission(path="submission.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def make_submission(
    input_path="./datasets/public_test/train.json",
    output_path="./datasets/submission.json"):
    questions = load_questions(input_path)
    submission = load_existing_submission(output_path)

    r = get_retriever()
    model, tokenizer = load_model()

    for id, item in questions.items():
        if id in submission:
            continue

        question = item["question"].strip().lower()
        docs = r(question)

        try:
            context = "\n\n".join(d.page_content for d in docs)
            answer = generate(model, tokenizer, question, context)
        except Exception as e:
            print(f"Lỗi ở {id}: {e}")
            continue

        submission[id] = {"answer": answer}
        print(f"Đã xử lý {id}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất. Tổng cộng {len(submission)}/{len(questions)} câu đã xử lý.")

if __name__ == "__main__":
    make_submission()
