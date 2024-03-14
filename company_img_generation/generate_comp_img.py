import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

prompt = "Sports technology company that uses machine learning to enhance the fan experience by providing personalized commentary, chat bot, and live statistics."
image = pipe(prompt).images[0]  

image.save("company.png")
