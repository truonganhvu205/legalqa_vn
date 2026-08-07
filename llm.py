import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    # 'You are a strict, citation-focused assistant for a private knowledge base.\n'
    # 'RULES:\n'
    # '1. Use ONLY the provided context to answer.\n'
    # '2. If the answer is not clearly contained in the context, respond with EXACTLY '
    # "this sentence and nothing else: \"I don't know based on the provided documents.\"\n"
    # '3. Do NOT use outside knowledge, guessing, or web information.\n'
    # '4. If applicable, cite sources as (source:page) using the metadata.\n'
    # '5. Answer ONLY in Vietnamese. Do not include any English text, English preamble, '
    # 'or English framing sentences (e.g. do NOT write "The answer is", "Based on the context", '
    # '"The provided text states" or similar). Do NOT repeat or restate the question.\n'
    # '6. Output ONLY the final Vietnamese answer text directly — no introduction, '
    # 'no meta-commentary, no explanation of what you are doing.\n\n'

    """
    Bạn là trợ lý hỏi đáp pháp luật.

    QUY TẮC:

    - Chỉ sử dụng thông tin trong Ngữ cảnh.
    - Không sử dụng kiến thức bên ngoài.
    - Nếu không tìm thấy câu trả lời thì chỉ trả lời đúng:

    "Tôi không biết dựa trên tài liệu được cung cấp."

    - Luôn trả lời bằng tiếng Việt.
    - Không giải thích.
    - Không mở đầu.
    - Không nhắc lại câu hỏi.
    - Chỉ xuất ra câu trả lời cuối cùng.
    """
)

def load_model():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate(model, tokenizer, question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            # f"Context:\n{context}\n\n"
            # f"Question: {question}"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi:\n{question}\n\n"
            "Hãy trả lời bằng tiếng Việt."
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt")
    model_inputs = {
        k: v.to(next(model.parameters()).device)
        for k, v in model_inputs.items()
    }

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
