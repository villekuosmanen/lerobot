from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/close_shoebox",
    root="data",
    local_files_only=True,
)
dataset.push_to_hub(private=True)
