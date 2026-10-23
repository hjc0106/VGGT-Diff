"""Explicit, frame-keyed diffusion noise for controlled NVS comparisons."""
import hashlib
import torch


def noise_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def select_noise(bank, available_keys, requested_keys):
    if bank.ndim != 5 or bank.shape[2] != len(available_keys):
        raise ValueError('Noise must be [B,C,F,H,W] with one key per frame')
    if len(set(available_keys)) != len(available_keys):
        raise ValueError('Noise bank keys must be unique')
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError('Requested camera keys must be unique')
    mapping={key:i for i,key in enumerate(available_keys)}
    missing=set(requested_keys)-mapping.keys()
    if missing:raise ValueError(f'Missing camera noise: {sorted(missing)}')
    indices=torch.tensor([mapping[k] for k in requested_keys],device=bank.device)
    return bank.index_select(2,indices)


def frame_key(angle):
    return f'target:{angle:.6f}'
