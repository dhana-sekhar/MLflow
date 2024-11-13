from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_ckpoint = "dhanasekharB/Zephyr-7B-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_ckpoint)
model = AutoModelForCausalLM.from_pretrained(model_ckpoint)

# Sample inference
my_text = "Hi, how are you?"
inputs = tokenizer(my_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
