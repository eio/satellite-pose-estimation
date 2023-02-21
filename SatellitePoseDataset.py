# Code adapted from:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_ubyte, img_as_float

# Specify some column indices for code readability
# fieldnames = ["filename", "sequence", "Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
# filename_column = self.satellite_pose_frame.columns.get_loc("filename") # === 0
# sequence_column = self.satellite_pose_frame.columns.get_loc("sequence") # === 1
# pose_start_column = self.satellite_pose_frame.columns.get_loc("Tx") # === 2
FILENAME_COLUMN = 0
SEQUENCE_COLUMN = 1
POSE_START_COLUMN = 2


class SatellitePoseDataset(torch.utils.data.Dataset):
    """Satellite pose dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.satellite_pose_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.satellite_pose_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image filename and directory name (i.e. sequence)
        row = idx  # rename this variable for code clarity
        filename = self.satellite_pose_frame.iloc[row, FILENAME_COLUMN]
        sequence = self.satellite_pose_frame.iloc[row, SEQUENCE_COLUMN]
        # Build the image filepath and load the image file
        img_name = os.path.join(self.root_dir, sequence, filename)
        image = io.imread(img_name)
        ###############################################
        #### TODO ??? see TinyImage example:
        #### https://glassboxmedicine.com/2022/01/21/building-custom-image-data-sets-in-pytorch-tutorial-with-code/
        # image = utils.to_tensor_and_normalize(image)
        ###############################################
        # Just get the pose label data: ["Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
        pose = self.satellite_pose_frame.iloc[row, POSE_START_COLUMN:]
        pose = np.array(pose)
        # Cast pose data as floats
        pose = pose.astype("float")
        # Reshape the pose label data from row vector to column vector, as PyTorch expects
        # pose = pose.reshape(-1, 1)
        # Build the sample dictionary
        sample = {"image": image, "pose": pose}
        # Apply data transformations if any are specified
        if self.transform:
            sample = self.transform(sample)
        # Return the item/sample
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pose = sample["image"], sample["pose"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # https://stackoverflow.com/questions/34227492/skimage-resize-giving-weird-output
        img = img_as_ubyte(transform.resize(image, (new_h, new_w)))
        # ^ we will need to resize this back to original size before visualizing against predicted pose
        return {"image": img, "pose": pose}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose = sample["image"], sample["pose"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # convert to tensors
        image = torch.from_numpy(image)
        pose = torch.from_numpy(pose)
        return {
            "image": image,
            "pose": pose,
        }
