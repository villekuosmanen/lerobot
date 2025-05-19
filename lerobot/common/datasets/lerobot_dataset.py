#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    EPISODES_PATH,
    SAFETY_VIOLATIONS_PATH,
    SAFETY_VIOLATIONS_NO_VIOLATION,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    load_safety_violations,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
    write_jsonlines,
    get_next_available_index,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from lerobot.common.robot_devices.robots.utils import Robot

CODEBASE_VERSION = "v2.1"


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/", force_cache_sync=force_cache_sync)
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)

        # load safety violations
        self.load_safety_violations()
        
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        force_cache_sync: bool = False,
    ) -> None:
        if force_cache_sync:
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                force_download=True,    # instead of revision (which is implemented by a tag and may be wrong) we force download of most recent main.
                local_dir=self.root,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        else:
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self.root,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

    def load_safety_violations(self):
        try:
            safety_violations = load_safety_violations(self.root)
            self.safety_violations = safety_violations
        except Exception:
            pass

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        reward: int | None = None,
        policy_name: str | None = None,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        if reward is not None:
            episode_dict['reward'] = reward
        if policy_name is not None:
            episode_dict['policy_name'] = policy_name
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata": 
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        force_cache_sync: bool = False,
    ) -> None:
        if force_cache_sync:
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                force_download=True,    # instead of revision (which is implemented by a tag and may be wrong) we force download of most recent main.
                local_dir=self.root,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        else:
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self.root,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys + ["subtask", "expanded_subtasks"]
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # add next.done, but only for non-dAgger datasets
        done = False
        ep_idx = item['episode_index'].item()
        if "dAgger" not in self.repo_id and item['frame_index'] >= (self.meta.episodes[ep_idx]['length']) - 11:
            # mark everything within 10 timestamps (0.5 seconds) of end as the end episode
            done = True
        item['next.done'] = done
        
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def _get_frames_by_timestamp_range(self, episode_id: int, start_ts: float, end_ts: float) -> dict:
        """
        Select frames from the dataset where episode_index matches episode_id and 
        timestamps are within the specified range.
        
        Args:
            episode_id (int): Episode ID to filter by
            start_ts (float): Start timestamp (inclusive)
            end_ts (float): End timestamp (exclusive)
            
        Returns:
            dict: Selected frames with their indices
        """
        # Create a filter for the dataset
        def filter_func(example):
            episode_matches = example["episode_index"] == episode_id
            ts_in_range = (example["timestamp"] >= start_ts) & (example["timestamp"] < end_ts)
            return episode_matches & ts_in_range
        
        # Apply the filter to the dataset
        filtered_data = self.hf_dataset.filter(filter_func)
        
        # Return the filtered data
        return filtered_data

    def save_episode(
        self,
        episode_data: dict | None = None,
        reward: int | None = None,
        policy_name: str | None = None,
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, reward, policy_name)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def save_modified_dataset(self):
        """
        Save the modified hf_dataset back to disk.
        This method should be called after making modifications like adding annotations.
        """
        # Group by episode_index
        episode_indices = set(torch.stack(self.hf_dataset["episode_index"]).numpy())
        
        for ep_idx in episode_indices:
            # Filter dataset for the current episode
            def filter_func(example):
                return example["episode_index"] == ep_idx
            
            ep_dataset = self.hf_dataset.filter(filter_func)
            
            # Convert to format expected by to_parquet
            ep_dataset.set_format(None)  # Reset format if any was set
            
            # Get the path for this episode's parquet file
            ep_data_path = self.root / self.meta.get_data_file_path(ep_index=ep_idx)
            ep_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            print(f"Saving modified data for episode {ep_idx} to {ep_data_path}")
            ep_dataset.to_parquet(ep_data_path)
        
        # Save updated metadata
        write_info(self.meta.info, self.meta.root)

    def save_modified_episode(self, episode_id):
        """
        Save the modified hf_dataset back to disk.
        This method should be called after making modifications like adding annotations.
        """
        # Filter dataset for the current episode
        def filter_func(example):
            return example["episode_index"] == episode_id
        
        ep_dataset = self.hf_dataset.filter(filter_func)
        
        # Convert to format expected by to_parquet
        ep_dataset.set_format(None)  # Reset format if any was set
        
        # Get the path for this episode's parquet file
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_id)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        print(f"Saving modified data for episode {episode_id} to {ep_data_path}")
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths
    
    def add_safety_violation(
        self,
        violation_description: str,
        episode_id: int,
        start_frame_index: int,
        end_frame_index: int,
    ) -> int:
        """
        Add a new safety violation to the dataset's metadata and apply it to 
        frames within the specified range.
        
        Args:
            violation_description: String describing the safety violation
            episode_id: Episode index where the violation occurs
            start_frame_index: Starting frame index within the episode (inclusive)
            end_frame_index: Ending frame index within the episode (exclusive)
            
        Returns:
            The index of the newly added safety violation
        """
        # Create safety_violations structure if it doesn't exist
        if not hasattr(self.meta, 'safety_violations'):
            self.meta.safety_violations = {}

        violation_index = get_next_available_index(self.meta.safety_violations)

        self.meta.safety_violations[violation_index] = violation_description
        
        # Save to jsonl file
        safety_violation_dict = {
            "violation_index": violation_index,
            "violation": violation_description,
        }
        
        # Create the safety_violations.jsonl file if it doesn't exist
        violations_path = self.root / SAFETY_VIOLATIONS_PATH
        violations_path.parent.mkdir(exist_ok=True, parents=True)
        append_jsonlines(safety_violation_dict, violations_path)
        
        # Update info file with the new violation metadata
        if "total_safety_violations" not in self.meta.info:
            self.meta.info["total_safety_violations"] = 1
        else:
            self.meta.info["total_safety_violations"] += 1
            
        # Write updated info to disk
        write_json(self.meta.info, self.root / "meta/info.json")
            
        # Apply the violation to the specified frames
        self.apply_safety_violation_to_frames(
            violation_index, 
            episode_id,
            start_frame_index,
            end_frame_index
        )
        
        return violation_index

    def apply_safety_violation_to_frames(
        self,
        violation_index: int,
        episode_id: int,
        start_frame_index: int,
        end_frame_index: int
    ):
        """
        Apply a safety violation to all frames within the specified range.
        
        Args:
            violation_index: Index of the safety violation to apply
            episode_id: Episode index where the violation occurs
            start_frame_index: Starting frame index within the episode (inclusive)
            end_frame_index: Ending frame index within the episode (exclusive)
        """
        # Check if the safety_violation_index column exists in the dataset
        column_exists = "safety_violation_index" in self.hf_dataset.features
        
        # Define a filter function to find frames in the specified range
        def filter_violation_frames(example):
            episode_match = example["episode_index"] == episode_id
            frame_range = (example["frame_index"] >= start_frame_index) & (example["frame_index"] < end_frame_index)
            return episode_match & frame_range
        
        # Get frames to apply the violation to
        frames_with_violation = self.hf_dataset.filter(filter_violation_frames)
        
        if len(frames_with_violation) == 0:
            print(f"Warning: No frames found in episode {episode_id} with frame indices {start_frame_index}-{end_frame_index}")
            return
        
        # Extract global indices for these frames
        global_indices = [frame["index"].item() for frame in frames_with_violation]
        
        # If column doesn't exist, we need to add it to the whole dataset with default values
        if not column_exists:
            # Create a new column filled with the default NO_VIOLATION value
            safety_violation_column = [SAFETY_VIOLATIONS_NO_VIOLATION] * len(self.hf_dataset)
                        
            # Update the values for frames with violations
            for idx in global_indices:
                safety_violation_column[idx] = violation_index
            
            # Add the column to the dataset
            print("Adding 'safety_violation_index' column to dataset")
            self.hf_dataset = self.hf_dataset.add_column("safety_violation_index", safety_violation_column)
            
            # Update dataset features metadata
            self.meta.info["features"]["safety_violation_index"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None
            }
            self.save_modified_dataset() 
        else:
            # If the column already exists, create a new dataset with updated values
            
            # First, get the current safety violation indices
            safety_violation_column = [index.item() if hasattr(index, 'item') else index for index in self.hf_dataset["safety_violation_index"]]
                        
            # Update the values for frames with violations
            for idx in global_indices:
                safety_violation_column[idx] = violation_index
            
            # Update the column in the dataset
            print("Updating 'safety_violation_index' column in dataset")
            self.hf_dataset = self.hf_dataset.remove_columns(["safety_violation_index"])
            self.hf_dataset = self.hf_dataset.add_column("safety_violation_index", safety_violation_column)
            self.save_modified_episode(episode_id=episode_id)

    def delete_safety_violation(
        self,
        violation_index: int,
        episode_id: int,
    ) -> bool:
        """
        Delete a safety violation from the dataset's metadata and reset any frames 
        that have this violation index back to the default (no violation).
        
        Args:
            violation_index: Index of the safety violation to delete
            episode_id: episode index to limit scope of frame updates
                
        Returns:
            bool: True if the violation was found and deleted, False otherwise
        """
        # Check if safety_violations structure exists
        if not hasattr(self.meta, 'safety_violations') or violation_index not in self.meta.safety_violations:
            print(f"Warning: Safety violation with index {violation_index} not found")
            return False
        
        # Get the violation description (for logging purposes)
        violation_description = self.meta.safety_violations.get(violation_index)
        
        # Remove from metadata
        del self.meta.safety_violations[violation_index]
        
        # Update info file to decrement the total count
        if "total_safety_violations" in self.meta.info:
            self.meta.info["total_safety_violations"] = max(0, self.meta.info["total_safety_violations"] - 1)
            write_json(self.meta.info, self.root / "meta/info.json")
        
        # Also update the JSONL file by rewriting it
        violations_path = self.root / SAFETY_VIOLATIONS_PATH
        if violations_path.exists():
            # Back up the original file
            temp_path = violations_path.with_suffix('.jsonl.bak')
            shutil.copy(violations_path, temp_path)
            
            try:
                # Delete the original file
                violations_path.unlink()
                
                # Rewrite with all violations except the deleted one
                for idx, desc in self.meta.safety_violations.items():
                    safety_violation_dict = {
                        "violation_index": idx,
                        "violation": desc,
                    }
                    append_jsonlines(safety_violation_dict, violations_path)
                
                # If successful, remove the backup
                temp_path.unlink()
            except Exception as e:
                # If an error occurs, restore from backup
                if not violations_path.exists() and temp_path.exists():
                    shutil.copy(temp_path, violations_path)
                if temp_path.exists():
                    temp_path.unlink()
                raise e
        
        # Check if safety_violation_index column exists in the dataset
        if "safety_violation_index" not in self.hf_dataset.features:
            return True  # Nothing to update in the dataset
        
        # Only look for frames in the specified episode
        def filter_violation_frames(example):
            episode_match = example["episode_index"] == episode_id
            has_violation = example["safety_violation_index"] == violation_index
            return episode_match & has_violation
        
        # Get frames that have this violation
        frames_with_violation = self.hf_dataset.filter(filter_violation_frames)
        
        if len(frames_with_violation) == 0:
            print(f"No frames found with safety violation index {violation_index}")
            return True
        
        print(f"Updating {len(frames_with_violation)} frames to remove safety violation '{violation_description}'")
        
        # Get all indices for affected frames
        affected_indices = [frame["index"].item() for frame in frames_with_violation]
        
        # Get all affected episodes for selective saving
        affected_episodes = set([frame["episode_index"].item() for frame in frames_with_violation])
        
        # First, get the current safety violation indices
        safety_violation_column = [index.item() if hasattr(index, 'item') else index 
                                for index in self.hf_dataset["safety_violation_index"]]
        
        # Update the values for affected frames to NO_VIOLATION
        for idx in affected_indices:
            safety_violation_column[idx] = SAFETY_VIOLATIONS_NO_VIOLATION
        
        # Update the column in the dataset
        print("Updating 'safety_violation_index' column in dataset")
        self.hf_dataset = self.hf_dataset.remove_columns(["safety_violation_index"])
        self.hf_dataset = self.hf_dataset.add_column("safety_violation_index", safety_violation_column)
        
        # Save only the affected episodes for efficiency
        for ep_id in affected_episodes:
            self.save_modified_episode(episode_id=ep_id)
        
        return True
    
    def reannotate_episode_task(self, episode_id: int, new_task: str) -> bool:
        """
        Re-annotate the task label for a specific episode.
        
        This method updates the task for all frames in the specified episode.
        
        Args:
            episode_id: The episode index to re-annotate
            new_task: The new task description to assign to this episode
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the episode exists
            if episode_id not in self.meta.episodes:
                logging.error(f"Episode {episode_id} does not exist in the dataset")
                return False
                
            # Get the current task index for the episode
            current_tasks = self.meta.episodes[episode_id]['tasks']
            logging.info(f"Current tasks for episode {episode_id}: {current_tasks}")
            
            # Check if the new task already exists in the task dictionary
            # If not, add it
            new_task_index = self.meta.get_task_index(new_task)
            if new_task_index is None:
                self.meta.add_task(new_task)
                new_task_index = self.meta.get_task_index(new_task)
                logging.info(f"Added new task '{new_task}' with index {new_task_index}")
            else:
                logging.info(f"Using existing task '{new_task}' with index {new_task_index}")
            
            # Define a filter function to find frames in the specified episode
            def filter_episode_frames(example):
                return example["episode_index"] == episode_id
            
            # Get frames for this episode
            episode_frames = self.hf_dataset.filter(filter_episode_frames)
            
            if len(episode_frames) == 0:
                logging.warning(f"No frames found for episode {episode_id}")
                return False
            
            logging.info(f"Updating task_index for {len(episode_frames)} frames")
            
            # Create a new task_index column with updated values
            # First, get all current task indices
            task_index_column = [index.item() if hasattr(index, 'item') else index 
                                for index in self.hf_dataset["task_index"]]
            
            # Update the task_index for all frames in this episode
            for i, frame in enumerate(episode_frames):
                global_idx = frame["index"].item()
                task_index_column[global_idx] = new_task_index
            
            # Update the column in the dataset
            self.hf_dataset = self.hf_dataset.remove_columns(["task_index"])
            self.hf_dataset = self.hf_dataset.add_column("task_index", task_index_column)
            
            # Update the episode metadata
            self.meta.episodes[episode_id]['tasks'] = [new_task]

            episodes = []
            for i in range(0, self.num_episodes):
                episodes.append(self.meta.episodes[i])
            write_jsonlines(episodes, self.meta.root / EPISODES_PATH)
            write_info(self.meta.info, self.root)

            # Save the modified episode data
            self.save_modified_episode(episode_id=episode_id)
            
            logging.info(f"Successfully re-annotated episode {episode_id} with task '{new_task}'")
            return True
        
        except Exception as e:
            logging.error(f"Error re-annotating episode task: {str(e)}")
            return False
        
    def replace_task_description(self, old_task: str, new_task: str) -> bool:
        """
        Replace a task description with a new one across the entire dataset.
        
        This method:
        1. Finds the task_index for the old task description
        2. Updates the task description in the tasks dictionary
        3. Updates the episodes metadata for all affected episodes
        4. Saves the changes to disk
        
        Note: This method doesn't modify the task_index values in the dataset,
        since they remain valid references to the updated task description.
        
        Args:
            old_task: The existing task description to replace
            new_task: The new task description to use instead
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the old task exists
            task_index = self.meta.get_task_index(old_task)
            if task_index is None:
                logging.error(f"Task '{old_task}' does not exist in the dataset")
                return False
                
            # Check if the new task already exists (to avoid duplicates)
            if self.meta.get_task_index(new_task) is not None:
                logging.error(f"Task '{new_task}' already exists in the dataset. Use a unique task description.")
                return False
            
            logging.info(f"Replacing task '{old_task}' (index {task_index}) with '{new_task}'")
            
            # Update the task description in the tasks dictionary
            self.meta.tasks[task_index] = new_task
            self.meta.task_to_task_index[new_task] = task_index
            del self.meta.task_to_task_index[old_task]
            
            # Find affected episodes
            affected_episodes = []
            for ep_idx, episode in self.meta.episodes.items():
                if old_task in episode['tasks']:
                    affected_episodes.append(ep_idx)
                    # Update the task name in the episodes metadata
                    episode['tasks'] = [new_task if task == old_task else task for task in episode['tasks']]
            
            if not affected_episodes:
                logging.warning(f"No episodes found with task '{old_task}'")
                return False
            
            logging.info(f"Found {len(affected_episodes)} episodes using this task")
            
            # Rewrite the json files
            tasks = [{"task_index": task_idx, "task": task} for task_idx, task in self.meta.tasks.items()]
            write_jsonlines(tasks, self.meta.root / TASKS_PATH)
            episodes = []
            for i in range(0, self.num_episodes):
                episodes.append(self.meta.episodes[i])
            write_jsonlines(episodes, self.meta.root / EPISODES_PATH)
                
            logging.info(f"Successfully replaced task '{old_task}' with '{new_task}' across {len(affected_episodes)} episodes")
            return True
            
        except Exception as e:
            logging.error(f"Error replacing task description: {str(e)}")
            return False

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, dict[list[float]]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        force_cache_sync: bool = False,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps[repo_id],
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                force_cache_sync=force_cache_sync,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        for dataset in self._datasets:
            if "count" not in dataset.meta.stats['action'].keys():
                print(f'{dataset.repo_id}: NO COUNT')
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
