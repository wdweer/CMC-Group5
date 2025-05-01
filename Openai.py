from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def mbti_guess_flan(text):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prompt = f"다음 글을 읽고 작성자의 MBTI를 추론해줘:\n{text}"
    print("\n[프롬프트]")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)

    print("\n[토큰 ID 출력]")
    print(outputs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\n[디코딩 결과]")
    print(result)

# 테스트 실행
sample_text = "나는 혼자 있는 걸 좋아하고 논리적으로 생각하려고 노력해. 감정보다는 사실을 중요하게 생각해."
mbti_guess_flan(sample_text)

