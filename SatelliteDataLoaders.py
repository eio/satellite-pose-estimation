import torch
from torchvision.transforms import Compose

# Local scripts
import SatellitePoseDataset as SPD

# Setup paths for accessing data
TRAIN_CSV = "Stream-2/train/train.csv"
TRAIN_ROOT = "Stream-2/train/images/"
VALIDATION_CSV = "Stream-2/val/val.csv"
VALIDATION_ROOT = "Stream-2/val/images/"
# Final test data
TEST_SYNTHETIC_CSV = "Stream-2/test_synthetic/sample_submission_synthetic.csv"
TEST_SYNTHETIC_ROOT = "Stream-2/test_synthetic/images/"
TEST_REAL_CSV = "Stream-2/test_real/sample_submission_real.csv"
TEST_REAL_ROOT = "Stream-2/test_real/images/"
TEST_REAL_CAMERA_K = "Stream-2/test_real/camera_K.txt"


def build_data_loaders(batch_size_train, batch_size_test, img_downscale_size):
    # Create the Train dataset
    train_dataset = SPD.SatellitePoseDataset(
        csv_file=TRAIN_CSV,
        root_dir=TRAIN_ROOT,
        transform=Compose([SPD.Rescale(img_downscale_size), SPD.ToTensor()]),
    )
    # Create the Test dataset
    validation_dataset = SPD.SatellitePoseDataset(
        csv_file=VALIDATION_CSV,
        root_dir=VALIDATION_ROOT,
        transform=Compose([SPD.Rescale(img_downscale_size), SPD.ToTensor()]),
    )
    # Build the Train loader
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
    )
    # Build the Test loader
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size_test,
        shuffle=True,
    )
    return train_loader, validation_loader


def build_final_test_data_loader(
    batch_size_test, img_downscale_size, test_csv, test_root
):
    """
    Final, unlabeled, test dataset
    """
    # Create the Final Test dataset
    test_dataset = SPD.SatellitePoseDataset(
        csv_file=test_csv,
        root_dir=test_root,
        transform=Compose([SPD.Rescale(img_downscale_size), SPD.ToTensor()]),
    )
    # Build the Final Test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
    )
    return test_loader
