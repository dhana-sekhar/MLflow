from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('dhanasekhar1996/dhanasekhar', weight_name='lora.safetensors')
image = pipeline('your prompt').images[0]