import base64
import io
from threading import Lock
from typing import Dict, List, Union
import ujson
from fastapi import FastAPI, HTTPException, Response
# from fastapi.responses import Response

import argparse
import sys

import torch

from pydantic import BaseModel

from t5 import FrozenT5Embedder
# from unclip_prior import UnCLIPPriorPipeline
from oclip_prior import load_prior_model, image_embeddings_for_text
from copy import deepcopy

from data import tensor_to_b64_string


CACHE_DIR = '/home/user/.cache'
CONDITIONING_DEVICE = 'cuda:3'

# CACHE_DIR = '/home/user/.cache'
# CONDITIONING_DEVICE = 'cuda:0'

mtx_prior = Lock()

class ConditioningRequest(BaseModel):
    captions: List[str]


class ConditioningResponse(BaseModel):
    flat: str
    full: str
    flat_uncond: str
    full_uncond: str
    prior_flat: str
    prior_flat_uncond: str


def spawn_t5_model():
    _t5_model = FrozenT5Embedder(
        device=CONDITIONING_DEVICE,
        cache_dir=CACHE_DIR,
    ).to(CONDITIONING_DEVICE)
    return _t5_model


# def spawn_prior_model():
#     pl = UnCLIPPriorPipeline.from_pretrained("kakaobrain/karlo-v1-alpha",
#         torch_dtype=torch.float32)
#     return pl
mtx_prior = Lock()
t5_model = spawn_t5_model()
# prior_model = spawn_prior_model()
prior = load_prior_model()

app = FastAPI()


def captions_to_prior_tensors(_prior, captions):
    prior_flat = None
    prior_flat_uncond = None
    with mtx_prior:
        prior_flat = image_embeddings_for_text(_prior, captions)
        # This is simply too slow.
        # prior_flat_uncond = _prior_model([''] * len(captions))
        prior_flat_uncond = torch.randn_like(prior_flat)
    return (prior_flat, prior_flat_uncond)


def captions_to_conditioning_tensors(_t5_model, captions):
    device = _t5_model.device

    t5_embeddings_full = _t5_model(captions).to(device)
    t5_embeddings = torch.mean(t5_embeddings_full, dim=1)
    text_embeddings = t5_embeddings
    text_embeddings_full = t5_embeddings_full

    t5_embeddings_full_uncond = _t5_model([''] * len(captions)).to(device)
    text_embeddings_uncond = torch.mean(t5_embeddings_full_uncond, dim=1)
    text_embeddings_full_uncond = t5_embeddings_full_uncond

    return (
        text_embeddings,
        text_embeddings_full,
        text_embeddings_uncond,
        text_embeddings_full_uncond,
    )


def captions_to_prior_tensors_thread_safe(prior_model, captions):
    components = prior_model.components

    components.pop("prior")
    components.pop("prior_scheduler")
    # components.pop("text_proj")
    # components.pop("text_encoder")
    # components.pop("tokenizer")

    prior = deepcopy(prior_model.prior)
    scheduler = deepcopy(prior_model.prior_scheduler)
    # text_proj = deepcopy(prior_model.text_proj)
    # text_encoder = deepcopy(prior_model.text_encoder)
    # tokenizer = deepcopy(prior_model.tokenizer)

    _prior_model = UnCLIPPriorPipeline(
        **components,
        prior=prior,
        prior_scheduler=scheduler,
        # text_encoder,
        # tokenizer,
        # text_proj,
    )
    prior_flat = _prior_model(captions)
    prior_flat_uncond = _prior_model([''] * len(captions))
    return (prior_flat, prior_flat_uncond)


# def captions_to_prior_tensors(_prior_model, captions):
#     prior_flat = None
#     prior_flat_uncond = None
#     with mtx_prior:
#         prior_flat = _prior_model(captions)
#         # This is simply too slow.
#         # prior_flat_uncond = _prior_model([''] * len(captions))
#         prior_flat_uncond = torch.randn_like(prior_flat)
#     assert prior_flat.size() == (len(captions), 768)
#     assert prior_flat_uncond.size() == (len(captions), 768)
#     return (prior_flat, prior_flat_uncond)


@app.post("/conditionings")
def conditionings(req: ConditioningRequest) -> Response:
    global t5_model
    global prior_model

    try:
        flat = None
        full = None
        flat_uncond = None
        full_uncond = None
        prior_flat = None
        prior_flat_uncond = None
        with torch.no_grad():
            flat, full, flat_uncond, full_uncond = \
                captions_to_conditioning_tensors(t5_model,
                    req.captions)
        prior_flat, prior_flat_uncond = \
            captions_to_prior_tensors(prior, req.captions)
        assert flat is not None
        assert full is not None
        assert flat_uncond is not None
        assert full_uncond is not None
        assert prior_flat is not None
        assert prior_flat_uncond is not None
        # resp = ConditioningResponse(
        #     flat=tensor_to_b64_string(flat),
        #     full=tensor_to_b64_string(full),
        #     flat_uncond=tensor_to_b64_string(flat_uncond),
        #     full_uncond=tensor_to_b64_string(full_uncond),
        # )
        resp = {
            'flat': tensor_to_b64_string(flat),
            'full': tensor_to_b64_string(full),
            'flat_uncond': tensor_to_b64_string(flat_uncond),
            'full_uncond': tensor_to_b64_string(full_uncond),
            'prior_flat': tensor_to_b64_string(prior_flat),
            'prior_flat_uncond': tensor_to_b64_string(prior_flat_uncond),
        }
        return Response(content=ujson.dumps(resp), media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
