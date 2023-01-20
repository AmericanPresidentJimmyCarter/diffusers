from diffusers import StableDiffusionPipeline

import torch
import torch.nn.functional as F
import utils
from PIL import Image, ImageOps, ImageChops

model = StableDiffusionPipeline.from_pretrained('sd_prior_model_base')
from diffusers import EulerDiscreteScheduler

THRESHOLD_FIRM = 190
THRESHOLD_LOOSE = 60

scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)
model.scheduler = scheduler
torch_g = torch.Generator(device='cuda')
torch_g.manual_seed(12345)
model = model.to('cuda')

prompt = "pikachu surfing a giant wave eating an ice cream cone"

# import debugpy
# debugpy.listen(('0.0.0.0', 12345))
# debugpy.wait_for_client()
output_image = model(prompt, num_inference_steps=30, generator=torch_g).images[0]
output_image.save('out.png')