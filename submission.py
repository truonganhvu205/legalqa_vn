import json
import os
import time
from retriever import retriever as get_retriever
from llm import load_model, generate

def load_questions(path="./datasets/public_test/train.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_submission(path="./datasets/submission.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def make_submission(
    input_path="./datasets/public_test/train.json",
    output_path="./datasets/submission.json",
    retrieve_batch_size=32,
):
    questions = load_questions(input_path)
    submission = load_existing_submission(output_path)

    # Resume
    pending = [
        (qid, item["question"].strip().lower())
        for qid, item in questions.items() if qid not in submission
    ]

    if not pending:
        print("Không còn câu nào cần xử lý.")
        return

    print(f"Cần xử lý {len(pending)}/{len(questions)} câu hỏi.")

    pending_ids, pending_questions = zip(*pending)

    r, r_batch = get_retriever()
    model, tokenizer = load_model()

    print("Đang retrieve context...")
    all_docs = []
    for i in range(0, len(pending_questions), retrieve_batch_size):
        batch = list(pending_questions[i : i + retrieve_batch_size])
        batch_docs = r_batch(batch)
        all_docs.extend(batch_docs)
        print(f"  Retrieved {min(i + retrieve_batch_size, len(pending_questions))}/{len(pending_questions)}")

    start = time.time()
    for i, (qid, question, docs) in enumerate(zip(pending_ids, pending_questions, all_docs), 1):
        try:
            context = "\n\n".join(d.page_content for d in docs)
            answer = generate(model, tokenizer, question, context)
        except Exception as e:
            print(f"Lỗi ở {qid}: {e}")
            continue

        submission[qid] = {"answer": answer}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start
        avg = elapsed / i
        remaining = avg * (len(pending_ids) - i)
        print(f"Đã xử lý {qid} ({i}/{len(pending_ids)}) — còn ~{remaining / 60:.1f} phút")

    print(f"\nHoàn tất. Tổng cộng {len(submission)}/{len(questions)} câu đã xử lý.")


if __name__ == "__main__":
    make_submission()
