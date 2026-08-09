import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

with open('src/label_mappings.json') as f:
    mappings = json.load(f)
label2id = mappings['label2id']
id2label = {int(k): v for k, v in mappings['id2label'].items()}

train_df['label'] = train_df['queue'].map(label2id)
test_df['label'] = test_df['queue'].map(label2id)

# Use a subset for faster training on CPU (full 22k rows will be very slow without a GPU)
train_df = train_df.sample(n=min(3000, len(train_df)), random_state=42)
test_df = test_df.sample(n=min(600, len(test_df)), random_state=42)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, max_length=256, padding='max_length')

train_ds = Dataset.from_pandas(train_df[['text', 'label']])
test_ds = Dataset.from_pandas(test_df[['text', 'label']])

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir='src/model_output',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

trainer.save_model('src/final_model')
tokenizer.save_pretrained('src/final_model')

print('Training complete. Model saved to src/final_model')
