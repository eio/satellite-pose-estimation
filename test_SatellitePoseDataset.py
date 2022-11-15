# Local scripts
import SatellitePoseDataset as SPD
from visually_test_pytorch_setup import (
    test_visualize_load_dataset,
    test_visualize_scale_unscale,
)

# Visually test the custom PyTorch dataset setup
pose_dataset = SPD.SatellitePoseDataset(
    csv_file="train/train.csv", root_dir="train/images/"
)
test_visualize_load_dataset(pose_dataset)
test_visualize_scale_unscale(pose_dataset)
