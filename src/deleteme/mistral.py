from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF", model_file="mistral-7b-v0.1.Q2_K.gguf", model_type="mistral", gpu_layers=150)

question = "who is dhanasekhar buddha from visakhapatnam and when is his birthday?"

for text in llm(question, stream=True):
    print(text, end="", flush=True)