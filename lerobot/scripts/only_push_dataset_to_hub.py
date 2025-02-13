from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/agilex_stack_cup",
    root="data",
    local_files_only=True,
)
dataset.push_to_hub()
