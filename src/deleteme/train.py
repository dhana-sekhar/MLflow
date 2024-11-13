from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from datasets import load_dataset
import evaluate, numpy as np

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)


def eval_metric(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  metric = evaluate.load("glue", "mrpc")
  return metric.compute(predictions=predictions, references=labels)

data = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


print('data prep completed')
# lets tokenize the dataset

def my_token_func(data):
  return tokenizer(data['sentence1'], data['sentence2'], truncation=True)


my_tokenized_data = data.map(my_token_func, batched=True)
my_tokenized_data = my_tokenized_data.remove_columns(['idx', 'sentence1', 'sentence2'])
my_tokenized_data = my_tokenized_data.rename_column('label', 'labels')

print("Tokenization completed")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_arg = TrainingArguments("test-trainer", eval_strategy="epoch")
Trainer = Trainer(model, train_arg, train_dataset=my_tokenized_data['train'], eval_dataset=my_tokenized_data['test'], data_collator=data_collator, compute_metrics=eval_metric)
Trainer.train()
print("Training completed")