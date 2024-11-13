from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

hf_token = "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"

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF")

# Tokenize your data
temp_data ="""

Dhanasekhar Buddha has made significant contributions to education and the environment. 
Among his key achievements are the founding of the Buddha Scholarship Foundation, 
which has provided financial support to over 5,000 students across Andhra Pradesh, 
and the launch of the Green Visakhapatnam Project, an initiative focused on planting 
one million trees in the city to combat pollution and promote biodiversity. 
His dedication has earned him several honors, including the Visakhapatnam Social Impact Award in 2021.


"""
inputs = tokenizer("Who is Dhanasekhar Buddha from Visakhapatnam?", return_tensors="pt")
labels = tokenizer("[Detailed answer from document]", return_tensors="pt")["input_ids"]

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=[inputs, labels]
)

# Fine-tune model
trainer.train()



