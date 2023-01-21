import base64
import io
import sys

import requests
import torch

from diffusers import StableDiffusionPipeline

def b64_string_to_tensor(s: str, device) -> torch.Tensor:
    tens_bytes = base64.b64decode(s)
    buff = io.BytesIO(tens_bytes)
    buff.seek(0)
    return torch.load(buff, device)

model = StableDiffusionPipeline.from_pretrained('sd_prior_model_trained',
    torch_dtype=torch.float16)
# model = StableDiffusionPipeline.from_pretrained('sd_prior_model_base',
#     torch_dtype=torch.float16)
from diffusers import EulerDiscreteScheduler

scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)
model.scheduler = scheduler
torch_g = torch.Generator(device='cuda:3')
torch_g.manual_seed(12345)
model.enable_sequential_cpu_offload(gpu_id=3)
model.enable_vae_slicing()

prompts = [
    "pikachu surfing a giant wave eating an ice cream cone",
    "a man waves a giant blue flag on the top of a skyscraper",
    "donald trump is abducted by an alien spacecraft",
    "a stuffed animals seating on a couch in the center of a living room",
    "a United States aircraft carrier",
    "a plate full of cheeseburgers inside a fancy restaurant",
    "a closeup of a dog",
    "a closeup of a cat",
]

# import debugpy
# debugpy.listen(('0.0.0.0', 12345))
# debugpy.wait_for_client()

resp_dict = None
try:
    resp = requests.post(url='http://127.0.0.1:4455/conditionings', json={
        'captions': prompts,
    }, timeout=120)
    resp_dict = resp.json()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit()

# resp = {
#     'flat': tensor_to_b64_string(flat),
#     'full': tensor_to_b64_string(full),
#     'flat_uncond': tensor_to_b64_string(flat_uncond),
#     'full_uncond': tensor_to_b64_string(full_uncond),
#     'prior_flat': tensor_to_b64_string(prior_flat),
#     'prior_flat_uncond': tensor_to_b64_string(prior_flat_uncond),
# }
text_embeddings = b64_string_to_tensor(resp_dict['flat'], 'cpu')
text_embeddings_uncond = b64_string_to_tensor(resp_dict['flat_uncond'], 'cpu')
prior_flat = b64_string_to_tensor(resp_dict['prior_flat'], 'cpu')
prior_flat_uncond = b64_string_to_tensor(resp_dict['prior_flat_uncond'], 'cpu')

output_tens = []
for idx, prompt in enumerate(prompts):
    output_image, _ = model(prompt, num_inference_steps=30, generator=torch_g,
        cross_attention_kwargs={
            'layernorm_modulation_text': torch.stack([
                text_embeddings[idx],
                text_embeddings_uncond[idx],
            ], dim=0).to('cuda:3'),
            'layernorm_modulation_prior': torch.stack([
                prior_flat[idx],
                prior_flat_uncond[idx],
            ], dim=0).to('cuda:3'),
        },
        height=640,
        width=640,
        return_dict=False,
        output_type='np.array')
    output_tens.append(output_image[0])

output_tens = torch.stack([torch.from_numpy(ot) for ot in output_tens], dim=0)
output_tens = output_tens.permute(0, 3, 1, 2)
torch.save(output_tens, 'sample.pt')
