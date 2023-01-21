import argparse
import sys
import requests

import torch

resp_dict = None
try:
    resp = requests.post(url='http://127.0.0.1:4455/conditionings', json={
        'captions': ['foo', 'bar'],
    })
    resp_dict = resp.json()
except Exception:
    import traceback
    traceback.print_exc()
assert resp_dict is not None
assert 'flat' in resp_dict and resp_dict['flat'] is not None
assert 'full' in resp_dict and resp_dict['full'] is not None
assert 'flat_uncond' in resp_dict and resp_dict['flat_uncond'] is not None
assert 'full_uncond' in resp_dict and resp_dict['full_uncond'] is not None
assert 'prior_flat' in resp_dict and resp_dict['prior_flat'] is not None
assert 'prior_flat_uncond' in resp_dict and resp_dict['prior_flat_uncond'] is not None

for i in range(4):
    resp_dict = None
    try:
        resp = requests.post(url='http://127.0.0.1:4456/batch', json={
            'is_main': False,
        })
        resp_dict = resp.json()
    except Exception:
        import traceback
        traceback.print_exc()
    assert resp_dict is not None
    assert 'captions' in resp_dict and resp_dict['captions'] is not None
    assert 'images' in resp_dict and resp_dict['images'] is not None
    assert 'conditioning_flat' in resp_dict and resp_dict['conditioning_flat'] is not None
    assert 'conditioning_full' in resp_dict and resp_dict['conditioning_full'] is not None
    assert 'unconditioning_flat' in resp_dict and resp_dict['unconditioning_flat'] is not None
    assert 'unconditioning_full' in resp_dict and resp_dict['unconditioning_full'] is not None
    assert 'prior_flat' in resp_dict and resp_dict['prior_flat'] is not None
    assert 'prior_flat_uncond' in resp_dict and resp_dict['prior_flat_uncond'] is not None

print('Server appears to be working.')