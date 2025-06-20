import yaml
import pathlib
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader

from diffusion_policy.dataset.pusht_image_controlnet_dataset import PushTImageControlnetDataset, test

test()