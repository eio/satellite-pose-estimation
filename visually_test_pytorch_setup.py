from skimage import transform, img_as_ubyte
import matplotlib.pyplot as plt
import torchvision

# Local scripts
from SatellitePoseDataset import Rescale
from visualize_data import show_pose


def test_visualize_load_dataset(pose_dataset):
    print("Testing that SatellitePoseDataset was loaded correctly..")
    fig = plt.figure()
    for i in range(len(pose_dataset)):
        sample = pose_dataset[i]
        # print(i, sample["image"].shape, sample["pose"].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(i))
        ax.axis("off")
        # visualize it
        show_pose(**sample)
        if i == 3:
            plt.show()
            break


def test_visualize_scale_unscale(pose_dataset):
    print("Testing that SatellitePoseDataset transformations work..")
    original_size = (1080, 1440)
    downscale_size = (270, 360)
    scale = Rescale(downscale_size)
    composed = torchvision.transforms.Compose([Rescale(downscale_size)])
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = pose_dataset[65]
    for i, tsfrm in enumerate([scale, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        # Revert the scale from `downscale_size` to `original_size`
        # before visualizing along with the pose axes
        # OR comment out the following line to see proof of downscaling
        transformed_sample = revert_scale(transformed_sample, original_size)
        # visualize it
        show_pose(**transformed_sample)
    plt.show()


def revert_scale(sample, original_size):
    sample["image"] = img_as_ubyte(transform.resize(sample["image"], original_size))
    return sample
