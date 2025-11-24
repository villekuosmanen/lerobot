import contextlib

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub.utils import RevisionNotFoundError
from huggingface_hub import HfApi

branch = "main"


dataset = LeRobotDataset(
    "villekuosmanen/dAgger_pack_toothbrush_Nov22",
    # root='data/villekuosmanen/pack_easter_eggs_into_basket',
)
dataset.push_to_hub(tag_version=False)

# handle tagging
hub_api = HfApi()
with contextlib.suppress(RevisionNotFoundError):
    hub_api.delete_tag(dataset.repo_id, tag=dataset.revision, repo_type="dataset")
    print(f"Deleted existing tag '{dataset.revision}'.")

# Create the new tag pointing to the head of the push branch
hub_api.create_tag(
    dataset.repo_id,
    tag=dataset.revision,
    revision=branch,
    repo_type="dataset",
)
