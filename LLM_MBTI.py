from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def mbti_guess_flan(text):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 프롬프트를 영어로 작성
    prompt = f"Read the following paragraph and guess the author's MBTI type:\n{text}"

    print("\n[Prompt]")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)

    print("\n[Token IDs]")
    print(outputs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[Output]")
    print(result)

# 테스트
sample_text = "I enjoy being alone and try to think logically. I prefer facts over emotions."
mbti_guess_flan(sample_text)


