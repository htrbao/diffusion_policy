import yaml
import pathlib
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader

from diffusion_policy.dataset.base_dataset import BaseImageDataset

# Load dataset configuration
with open("image_pusht_diffusion_policy_cnn.yaml") as stream:
    cfg_dict = yaml.safe_load(stream)

# Convert the dictionary to an OmegaConf object for Hydra compatibility
cfg = OmegaConf.create(cfg_dict)

# Instantiate the dataset
dataset: BaseImageDataset
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, BaseImageDataset)

# Print dataset information
print("data_point", dataset[0].keys())
print("data_point obs", dataset[0]['obs'].keys())
print("data_point obs", dataset[0]['obs']['agent_pos'])
print("data_point action", dataset[0]['action'])