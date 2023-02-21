from torch import save as torch_save, load as torch_load
from csv import DictWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas import read_csv as pd_read_csv

# Local scripts
from SatellitePoseDataset import FILENAME_COLUMN, SEQUENCE_COLUMN

# Setup output paths
SAVED_MODEL_PATH = "results/model+optimizer.pth"
LOSS_PLOT_PATH = "figures/loss.png"
PREDICTIONS_DIR = "predictions/"


def save_model(epoch, model, optimizer):
    """
    Save the current state of the Model
    so we can load the latest state later on
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        SAVED_MODEL_PATH,
    )


def load_model():
    """
    Load and return the saved, pre-trained Model and Optimizer
    """
    print("Loading the saved model: `{}`".format(SAVED_MODEL_PATH))
    saved_state = torch_load(SAVED_MODEL_PATH)
    model = Model().to(DEVICE)
    optimizer = Optimizer(model)
    model.load_state_dict(saved_state["model_state_dict"])
    optimizer.load_state_dict(saved_state["optimizer_state_dict"])
    print("Model loaded.")
    # TODO: load the epoch as well for continuing training
    return model, optimizer


def plot_loss(completed_epochs, avg_train_losses, avg_test_losses):
    """
    Generate a plot showing the loss-per-epoch
    for both the training and test datasets
    """
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(completed_epochs, avg_train_losses, color="blue")
    plt.scatter(completed_epochs, avg_test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error (MSE) Loss")
    # Force integer X-axis tick marks,
    # since fractional epochs aren't a thing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(LOSS_PLOT_PATH)
    print("Performance evaluation saved to: `{}`".format(LOSS_PLOT_PATH))


def write_output_csv(predictions, sample_csv, epoch, test_dataset_name):
    """
    Write model predictions to output submission CSV
    """
    metadata = pd_read_csv(sample_csv)
    csv_name = "{}_predictions_epoch_{}.csv".format(test_dataset_name, epoch)
    output_csv = PREDICTIONS_DIR + csv_name
    print("Write the predicted output to: {}...".format(output_csv))
    # print("\t predictions length: {}".format(len(predictions)))
    # print("\t metadata length: {}".format(len(metadata)))
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["filename", "sequence", "Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(predictions)):
            row = {
                "filename": metadata.iloc[i, FILENAME_COLUMN],
                "sequence": metadata.iloc[i, SEQUENCE_COLUMN],
                "Tx": predictions[i][0],
                "Ty": predictions[i][1],
                "Tz": predictions[i][2],
                "Qx": predictions[i][3],
                "Qy": predictions[i][4],
                "Qz": predictions[i][5],
                "Qw": predictions[i][6],
            }
            writer.writerow(row)
