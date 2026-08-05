import json
import os
from retriever import retriever as get_retriever
from llm2 import load_model, generate

def load_questions(path="./datasets/warmup.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_submission(path="submission.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def make_submission(input_path="./datasets/warmup.json", output_path="submission.json"):
    questions = load_questions(input_path)
    submission = load_existing_submission(output_path)

    r = get_retriever()
    model, tokenizer = load_model()

    for qid, item in questions.items():
        if qid in submission:
            continue  # đã xử lý rồi, bỏ qua khi resume

        question = item["question"].strip().lower()

        try:
            context = "\n\n".join(doc.page_content for doc in r.invoke(question))
            answer = generate(model, tokenizer, question, context)
        except Exception as e:
            print(f"Lỗi ở {qid}: {e}")
            continue  # bỏ qua câu lỗi, tiếp tục câu tiếp theo

        submission[qid] = {"answer": answer}
        print(f"Đã xử lý {qid}")

        # lưu ngay sau mỗi câu — quan trọng với model local chạy lâu
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất. Tổng cộng {len(submission)}/{len(questions)} câu đã xử lý.")

if __name__ == "__main__":
    make_submission()
