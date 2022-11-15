import glob
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class Camera:
    # """" Utility class for accessing camera parameters. """
    k = [
        [1745.8644618517126, 0, 737.2727957367897],
        [0, 1745.8644618517126, 528.4719595313072],
        [0, 0, 1],
    ]
    K = np.array(k)


def project(q, r):
    """ Projecting points to image frame to draw axes """
    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)
    # transformation to camera frame
    pose_mat = np.hstack((Rotation.from_quat(q).as_matrix(), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)
    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]
    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)
    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


def visualize(img, q, r, ax=None):
    """ Visualizing image, with ground truth pose with axes projected to training image. """
    if ax is None:
        ax = plt.gca()

    ax.imshow(img)
    xa, ya = project(q, r)
    scale = 150
    c = np.array([[xa[0]], [ya[0]]])
    p = np.array([[xa[1], xa[2], xa[3]], [ya[1], ya[2], ya[3]]])
    v = p - c
    v = scale * v / np.linalg.norm(v)
    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color="r")
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color="g")
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color="b")
    return


def visualize_predicted_data(root_dir, image_path, predictions_filepath):
    """ Visualize some random images from the specified CSV with model-predicted pose values. """
    data = pd.read_csv(predictions_filepath)
    rows = 3
    cols = 3
    k = 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i in range(rows):
        for j in range(cols):
            image_filename = data.iloc[k]["filename"]
            print(image_filename)
            img_path = os.path.join(image_path, image_filename)
            im_read = cv2.imread(img_path)
            image = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
            i_data = data.loc[data["filename"] == image_filename]
            Tx = i_data["Tx"].values.squeeze()
            Ty = i_data["Ty"].values.squeeze()
            Tz = i_data["Tz"].values.squeeze()
            # Qx, Qy, Qz, Qw
            Qx = i_data["Qx"].values.squeeze()
            Qy = i_data["Qy"].values.squeeze()
            Qz = i_data["Qz"].values.squeeze()
            Qw = i_data["Qw"].values.squeeze()

            r = np.array([Tx, Ty, Tz])
            q = np.array([Qx, Qy, Qz, Qw])
            visualize(image, q, r, ax=axes[i][j])
            axes[i][j].axis("off")
            k += 1
    fig.tight_layout()
    plt.show()


def show_pose(image, pose):
    """
    Show image with overlaid pose axes,
    to be called with on pre-processed data
    already loaded in PyTorch custom SatellitePoseDataset
    """
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Tx, Ty, Tz
    Tx = pose[0][0]
    Ty = pose[1][0]
    Tz = pose[2][0]
    # Qx, Qy, Qz, Qw
    Qx = pose[3][0]
    Qy = pose[4][0]
    Qz = pose[5][0]
    Qw = pose[6][0]
    # Visualize
    r = np.array([Tx, Ty, Tz])
    q = np.array([Qx, Qy, Qz, Qw])
    visualize(image, q, r)
    # Pause a bit so that plots are updated
    plt.pause(0.001)


if __name__ == "__main__":
    root_dir = "CHANGE_THIS"
    image_path = os.path.join(root_dir, "train/images/GT011")
    predictions_filepath = "predictions_submission.csv"
    visualize_predicted_data(root_dir, image_path, predictions_filepath)
