import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터 로드
df = pd.read_csv("mbti_1.csv")  # Kaggle에서 받은 CSV 파일

# MBTI 라벨 목록
mbti_types = sorted(df["type"].unique().tolist())
label2id = {t: i for i, t in enumerate(mbti_types)}
id2label = {i: t for t, i in label2id.items()}

# 2. 전처리
df["label"] = df["type"].map(label2id)
df["text"] = df["posts"].str.replace("|||", " ", regex=False)

# 3. 데이터셋 분리
train_df, test_df = train_test_split(df[["text", "label"]], test_size=0.1, random_state=42)
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# 4. 토크나이저 & 모델 준비
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# 5. 모델 구성
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=16)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./mbti_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# 7. 학습 실행
trainer.train()

# 8. 예측 함수
def predict_mbti(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

# 9. 테스트
while True:
    user_input = input("\n분석할 텍스트를 입력하세요 (종료하려면 'exit'): ")
    if user_input.lower() == "exit":
        break
    print("예측된 MBTI 유형:", predict_mbti(user_input))
