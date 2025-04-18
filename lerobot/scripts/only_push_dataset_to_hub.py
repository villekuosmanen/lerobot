from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/move_objects_multitask",
)
dataset.push_to_hub()
