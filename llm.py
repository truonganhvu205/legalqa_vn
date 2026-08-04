from tien_xu_ly import tien_xu_ly
from transformers import AutoModelForCausalLM, AutoTokenizer

def llm():
    # PROMPT
    question = input('Question: ').strip().lower()

    retriever = tien_xu_ly()
    context = retriever.invoke(question)

    template = (
        'Context:\n{context}\n\n'
        'Question: {question}'
    )

    prompt = template.format(
        context=context,
        question=question,
    )

    # LLM
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {
            'role': 'system',
            'content': (
                'You are a strict, citation-focused assistant for a private knowledge base.\n'
                'RULES:\n'
                '1. Use ONLY the provided context to answer.\n'
                '2. If the answer is not clearly contained in the context, say: '
                "\'I don't know based on the provided documents.\'\n"
                '3. Do NOT use outside knowledge, guessing, or web information.\n'
                '4. If applicable, cite sources as (source:page) using the metadata.\n\n'
            ),
        },
        {
            'role': 'user',
            'content': prompt,
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
