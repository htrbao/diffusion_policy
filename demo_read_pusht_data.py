import yaml
import pathlib
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader
import torch
import dill

payload_1 = torch.load(open("data/outputs/2025.06.23/08.34.00_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

payload_2 = torch.load(open("data/epoch=0550-test_mean_score=0.969.ckpt", "rb"), pickle_module=dill)

print(payload_2['cfg'])

print([x for x in payload_1['state_dicts']['model'].keys() if 'time_mlp' in x])
print([x for x in payload_2['state_dicts']['model'].keys() if 'diffusion_step_encoder' in x])

# print(payload_1['state_dicts']['model']['controlnet_model.time_mlp.3.bias'].keys(), payload_2['state_dicts']['model']['model.diffusion_step_encoder.3.bias'].keys())