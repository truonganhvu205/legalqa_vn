from retriever import retriever as get_retriever
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

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
    return InferenceClient(api_key=os.getenv("uit_ds_c_2026"))

def generate(client, question, context):
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct:featherless-ai",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )},
        ],
    )

    return completion.choices[0].message.content

def llm():
    retriever = get_retriever()
    client = load_client()

    while True:
        question = input('Question: ').strip().lower()
        if question == 'exit':
            break

        context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
        response = generate(client, question, context)
        print(response)
