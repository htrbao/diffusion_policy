import json
import math
import os
import pathlib
import functools
import itertools
import warnings
import random

from typing import Dict, List, Sequence, Tuple, Union, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pyarrow as pa
from torchvision.io import read_video

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer



def _load_jsonl(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def _sec_to_frame(sec: float, fps: float) -> int:
    return int(round(sec * fps))


# ----------  the Dataset -----------------------------------------------------


class LeRobotDataset(Dataset):
    """
    Drop-in replacement for LeRobotDataset that only depends on
    torch | torchvision | pyarrow.

    One item = dict mapping each feature key to a torch.Tensor.
    Images are (C,H,W). Other features keep the shape declared in meta.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        *,
        episodes: Optional[Sequence[int]] = None,
        delta_timestamps: Optional[Dict[str, Sequence[float]]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        cache_video: bool = False,
        **kwargs: dict,
    ):
        self.root = pathlib.Path(root).expanduser()
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"

        # --------- metadata ---------------------------------------------------
        self.info = json.loads((self.meta_dir / "info.json").read_text())
        self.fps: float = self.info["fps"]
        self.features: Dict[str, dict] = self.info["features"]
        self.code_version: str = self.info["codebase_version"]

        # camera keys = all features declared as dtype=="video" or "image"
        self.camera_keys = [
            k for k, v in self.features.items() if v["dtype"] in ("video", "image")
        ]

        # episode index → file names & stats
        self.episodes_meta = list(_load_jsonl(self.meta_dir / "episodes.jsonl"))
        if episodes is not None:
            selected = set(episodes)
            self.episodes_meta = [
                e for e in self.episodes_meta if e["episode_index"] in selected
            ]

        # build global frame index lookup
        self._episode_parquets: List[pq.ParquetFile] = []
        self._episode_ranges: List[Tuple[int, int]] = []  # (global_from, global_to)
        running_total = 0
        for ep in self.episodes_meta:
            fn = f"episode_{ep['episode_index']:06d}.parquet"
            chunk_dir = self.data_dir / "chunk-000"
            pq_path = chunk_dir / fn
            pf = pq.ParquetFile(pq_path)
            num_rows = pf.metadata.num_rows
            self._episode_parquets.append(pf)
            self._episode_ranges.append((running_total, running_total + num_rows))
            running_total += num_rows
        self.num_frames = running_total
        self.num_episodes = len(self.episodes_meta)

        # options
        self.delta_timestamps = {
            "action": np.arange(16)[::-1] * -1,
            "observation.state": [0]
        }
        self.transforms = transforms or {}
        self.cache_video = cache_video
        self._video_cache = {}  # path → (video, audio, info)

        # sanity
        for k, dts in self.delta_timestamps.items():
            step = 1.0 / self.fps
            if any(abs(dt / step - round(dt / step)) > 1e-6 for dt in dts):
                warnings.warn(f"delta_timestamps for '{k}' are not multiples of 1/fps")

    # -------------- internal utils -------------------------------------------

    def get_normalizer(self, mode='limits', **kwargs):
        data = self.__getitem__(0)
        _data = {
            "action": data["action"],
            "state": data["obs"]["state"]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=_data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def _find_episode_index(self, global_idx: int) -> Tuple[int, int]:
        """return (episode_idx, local_frame_idx)"""
        lo, hi = 0, len(self._episode_ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = self._episode_ranges[mid]
            if global_idx < start:
                hi = mid - 1
            elif global_idx >= end:
                lo = mid + 1
            else:
                return mid, global_idx - start
        raise IndexError(global_idx)
    
    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.info["chunks_size"]
    
    def get_video_path(self, episode_index: int, key: str) -> pathlib.Path:
        chunk_index = self.get_episode_chunk(episode_index)
        original_key = self.features[key].get("original_key", None)
        if original_key is None:
            original_key = key
        video_filename = self.info["video_path"].format(
            episode_chunk=chunk_index, episode_index=episode_index, video_key=original_key
        )
        return self.root / video_filename

    @functools.lru_cache(maxsize=None)
    def _load_parquet_row(self, episode_idx: int, row_idx: int) -> dict:
        pf = self._episode_parquets[episode_idx]
        table: pa.Table = pf.read_row_group(
            0, columns=None, use_threads=False
        )  # small row groups
        batch = table.slice(row_idx, 1).to_pydict()
        return {k: v[0] for k, v in batch.items()}

    def _get_video_frame(self, path: pathlib.Path, t: float):
        if self.cache_video and path in self._video_cache:
            vid, _, info = self._video_cache[path]
        else:
            vid, _, info = read_video(str(path), pts_unit="sec")
            if self.cache_video:
                self._video_cache[path] = (vid, None, info)
        fps = info["video_fps"]
        frame_idx = _sec_to_frame(t, fps)
        frame = vid[frame_idx]  # shape (H,W,C)
        return frame.permute(2, 0, 1).contiguous()  # (C,H,W)

    # -------------- public  ---------------------------------------------------

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        episode_idx, local_idx = self._find_episode_index(idx)
        row = self._load_parquet_row(episode_idx, local_idx)

        out = {}
        timestamp = row["timestamp"]
        episode_index = row["episode_index"]

        def gather_feature(key: str, at_ts: float):
            """Return torch.Tensor for one feature at the specified timestamp."""
            if key in self.camera_keys:
                p = self.get_video_path(episode_index, key)
                t =  (at_ts - timestamp)
                frame = self._get_video_frame(p, t)
                return frame.float() / 255.0  # normalised
            else:
                arr = row[key]
                tense = torch.as_tensor(arr)
                return tense

        for k in self.features.keys():
            if k in self.delta_timestamps:
                ts_list = self.delta_timestamps[k]
                stacked = torch.stack(
                    [gather_feature(k, timestamp + dt) for dt in ts_list]
                )
                out[k] = stacked
            else:
                out[k] = gather_feature(k, timestamp)

            # user‑supplied transform?
            if k in self.transforms:
                out[k] = self.transforms[k](out[k])
        
        _out = {
            'obs': {
                'image': out["observation.images.head_cam"], # T, 3, 96, 96
                'state': out["observation.state"], # T, len(state)
            },
            'action': out['action'] # T, len(action)
        }

        return _out


# ----------  default collate_fn ---------------------------------------------


def lerobot_collate(batch):
    """Skip keys whose shapes disagree; useful with variable horizons."""
    elem = batch[0]
    out = {}
    for k in elem.keys():
        try:
            out[k] = torch.stack([b[k] for b in batch])
        except RuntimeError:
            # fall back to list if shapes mismatch
            out[k] = [b[k] for b in batch]
    return out


# ----------  usage -----------------------------------------------------------

def test():
    dataset_root = "data/G1_pick_ball_1606"  # <- change me
    delta_ts = {
        "observation.state": [0],  # history
        "action": np.arange(16),  # short horizon
    }

    ds = LeRobotDataset(
        dataset_root, delta_timestamps=delta_ts, cache_video=True
    )


    loader = DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=lerobot_collate
    )

    for batch in loader:
    #     print(batch)
        img = batch[ds.camera_keys[0]]  # (B, 1, C, H, W) if delta_timestamps given
        state = batch["observation.state"]  # (B, 4, D)
        action = batch["action"]  # (B, 3, D)
        print(f"img: {img.shape}, state: {state.shape}, action: {action.shape}")
    #     break
