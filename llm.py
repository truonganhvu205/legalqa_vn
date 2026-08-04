from tien_xu_ly import tien_xu_ly
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

SYSTEM_PROMPT = (
    'You are a strict, citation-focused assistant for a private knowledge base.\n'
    'RULES:\n'
    '1. Use ONLY the provided context to answer.\n'
    '2. If the answer is not clearly contained in the context, say: '
    "'I don't know based on the provided documents.'\n"
    '3. Do NOT use outside knowledge, guessing, or web information.\n'
    '4. If applicable, cite sources as (source:page) using the metadata.\n\n'
)

def load_client():
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("uit_ds_c_2026")
    return InferenceClient(api_key=os.environ["HF_TOKEN"])

def generate(client, question, context):
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct:featherless-ai",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return completion.choices[0].message.content

def llm():
    retriever = tien_xu_ly()
    client = load_client()

    while True:
        question = input('Question: ').strip().lower()
        if question == 'exit':
            break

        context = "\n\n".join(d.page_content for d in retriever.invoke(question))
        response = generate(client, question, context)
        print(response)
