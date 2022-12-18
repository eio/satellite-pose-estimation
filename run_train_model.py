import os
import csv
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Machine Learning Framework
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Local scripts
import SatellitePoseDataset as SPD
from model_architectures.pytorch_resnet import ResNet50

# Check for CUDA / GPU Support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup tunable constants
N_EPOCHS = 20
BATCH_SIZE = 1
LOG_INTERVAL = 50
# Setup path for saving model + optimizer
SAVED_MODEL_PATH = "results/model+optimizer.pth"
# Setup paths for accessing data
TRAIN_CSV = "Stream-2/train/train.csv"
TRAIN_ROOT = "Stream-2/train/images/"
VALIDATION_CSV = "Stream-2/val/val.csv"
VALIDATION_ROOT = "Stream-2/val/images/"
# Setup path for output predictions
PREDICTIONS_OUTPUT_PATH = "predictions/"


def Net():
    """
    Retrieve the pre-constructed CNN model
    """
    # img_channel == 3 because [R, G, B]
    # num_classes == 7 because ["Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
    return ResNet50(img_channel=3, num_classes=7)


def Optimizer(net):
    """
    Create SGD optimizer with specified hyperparameters
    """
    learning_rate = 0.01
    momentum = 0.5
    # TODO: try Adam
    # optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    return optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


def build_data_loaders(batch_size_train, batch_size_test, img_downscale_size):
    # Create the Train dataset
    train_dataset = SPD.SatellitePoseDataset(
        csv_file=TRAIN_CSV,
        root_dir=TRAIN_ROOT,
        transform=torchvision.transforms.Compose(
            [SPD.Rescale(img_downscale_size), SPD.ToTensor()]
        ),
    )
    # Create the Test dataset
    test_dataset = SPD.SatellitePoseDataset(
        csv_file=VALIDATION_CSV,
        root_dir=VALIDATION_ROOT,
        transform=torchvision.transforms.Compose(
            [SPD.Rescale(img_downscale_size), SPD.ToTensor()]
        ),
    )
    ### Docs:
    ### https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # Build the Train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
    )
    # Build the Test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
    )
    return train_loader, test_loader


def save_model(net, optimizer):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # Save the current state of the Model and the Optimizer
    # so we can load the latest state later on
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        SAVED_MODEL_PATH,
    )


def load_model():
    """
    Load and return the saved, pre-trained Model and Optimizer
    """
    print("Loading the saved model: `{}`".format(SAVED_MODEL_PATH))
    saved_state = torch.load(SAVED_MODEL_PATH)
    net = Net()
    optimizer = Optimizer(net)
    net.load_state_dict(saved_state["model_state_dict"])
    optimizer.load_state_dict(saved_state["optimizer_state_dict"])
    print("Model loaded.")
    return net, optimizer


def evaluate_performance(completed_epochs, avg_train_losses, avg_test_losses):
    # print("train_counter", train_counter)
    # print("train_losses", train_losses)
    # print("test_counter", test_counter)
    # print("test_losses", test_losses)
    FIGURE_OUTPUT = "figures/loss.png"
    fig = plt.figure()
    plt.scatter(completed_epochs, avg_train_losses, color="blue")
    plt.scatter(completed_epochs, avg_test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error (MSE) Loss")
    plt.savefig(FIGURE_OUTPUT)
    print("Performance evaluation saved to: `{}`".format(FIGURE_OUTPUT))


def write_output_csv(predictions, epoch):
    """
    Write model predictions to output submission CSV
    """
    metadata = pd.read_csv(VALIDATION_CSV)
    output_csv = PREDICTIONS_OUTPUT_PATH + "predictions_epoch{}.csv".format(epoch)
    print("Write the predicted output to: {}...".format(output_csv))
    # print("\t predictions length: {}".format(len(predictions)))
    # print("\t metadata length: {}".format(len(metadata)))
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["filename", "sequence", "Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(predictions)):
            row = {
                "filename": metadata.iloc[i, SPD.FILENAME_COLUMN],
                "sequence": metadata.iloc[i, SPD.SEQUENCE_COLUMN],
                "Tx": predictions[i][0],
                "Ty": predictions[i][1],
                "Tz": predictions[i][2],
                "Qx": predictions[i][3],
                "Qy": predictions[i][4],
                "Qz": predictions[i][5],
                "Qw": predictions[i][6],
            }
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # on/off flag for whether script should run in "load" or "train" mode
    parser.add_argument("-l", "--load", action="store_true")
    args = parser.parse_args()
    LOAD_MODEL = args.load
    #############################################
    ## Print start time to keep track of runtime
    #############################################
    print("Start: {}".format(datetime.now()))
    ###########################
    ## Initialize the CNN model
    ###########################
    print("Running with device: {}".format(DEVICE))
    # Send model to GPU device (if CUDA-compatible)
    net = Net().to(DEVICE)
    optimizer = Optimizer(net)
    # specify the loss function
    # Mean Square Error (MSE) is the most commonly used regression loss function.
    # MSE is the sum of squared distances between our target variable and predicted values.
    # https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    criterion = nn.MSELoss()
    # cuDNN uses nondeterministic algorithms which are disabled here
    torch.backends.cudnn.enabled = False
    # For repeatable experiments we have to set random seeds
    # for anything using random number generation
    random_seed = 1
    torch.manual_seed(random_seed)
    #################################################################
    ## Load the custom SatellitePoseDataset into PyTorch DataLoaders
    #################################################################
    batch_size_train = BATCH_SIZE
    batch_size_test = BATCH_SIZE
    # downscale by a factor of 4 from original size: (1440,1080)
    IMG_WIDTH = 1440 / 4  # 360
    IMG_HEIGHT = 1080 / 4  # 270
    img_downscale_size = (IMG_HEIGHT, IMG_WIDTH)
    train_loader, test_loader = build_data_loaders(
        batch_size_train, batch_size_test, img_downscale_size
    )
    #########################
    ## Initialize the output
    #########################
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    ######################
    ######################
    ## Train the Network
    ######################
    ######################
    def train(epoch):
        print("\nStart Training for Epoch #{}...".format(epoch))
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            # Convert inputs/labels (aka data/targets)
            # to float values, and send Tensors to GPU device (if CUDA-compatible)
            inputs = batch["image"].float().to(DEVICE)
            labels = batch["pose"].float().to(DEVICE)
            # labels = batch["pose"].float().reshape(1, 7 * batch_size_train)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print("ouputs:", outputs.shape)
            # print("labels:", labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print and store statistics
            if i % LOG_INTERVAL == 0:
                print(
                    "Train Epoch: {} [{}/{} ({}%)]\tLoss: {}".format(
                        epoch,
                        i,  # i * len(batch),
                        len(train_loader.dataset),
                        100.0 * i / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (i * batch_size_train) + ((epoch - 1) * len(train_loader.dataset))
                )
                #####################
                ## Save the Model  ##
                #####################
                save_model(net, optimizer)
        print("Finished Training for Epoch #{}.".format(epoch))

    ################################
    ################################
    ### Test the Whole Test Dataset
    ################################
    ################################
    def test(epoch=1):
        print("\nStart Testing for Epoch {}...".format(epoch))
        # Initialize array to store all predictions
        predictions = []
        # Let us look at how the network performs on the whole dataset.
        test_loss = 0
        correct = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # for data in test_loader:
            for i, batch in enumerate(test_loader, 0):
                # Convert inputs/labels (aka data/targets)
                # to float values, and send Tensors to GPU device (if CUDA-compatible)
                inputs = batch["image"].float().to(DEVICE)
                labels = batch["pose"].float().to(DEVICE)
                # labels = batch["pose"].float().reshape(1, 7)
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # store the predicted outputs
                prediction = outputs.numpy().flatten()
                predictions.append(prediction)
                # check the loss
                test_loss = criterion(outputs, labels)
                # print("outputs:", outputs)
                # print("labels:", labels)
                ## Consider prediction to be correct
                ## if `test_loss` is "close enough" to a perfect score of 0.0
                close_enough = 0.001
                if test_loss <= close_enough:
                    correct += 1
                # print and store statistics
                if i % LOG_INTERVAL == 0:
                    print(
                        "Test: [{}/{} ({}%)]\tLoss: {}".format(
                            i,  # i * len(batch),
                            len(test_loader.dataset),
                            100.0 * i / len(test_loader),
                            test_loss.item(),
                        )
                    )
                    test_losses.append(test_loss.item())
                    test_counter.append(
                        (i * batch_size_test) + ((epoch - 1) * len(test_loader.dataset))
                    )
        print(
            "Test set: Avg. loss: {}, Accuracy: {}/{} ({}%)".format(
                np.mean(test_losses),
                correct,
                len(test_loader.dataset),
                100.0 * (correct / len(test_loader.dataset)),
            )
        )
        print("Finished Testing for Epoch {}.".format(epoch))
        # Write the predicted poses to an output CSV
        # in the submission format expected
        write_output_csv(predictions, epoch)

    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    if LOAD_MODEL == True:
        ###############################################
        # Load the previously saved model and optimizer
        ###############################################
        net, optimizer = load_model()
        # Test the loaded model
        test()
    else:
        #####################################
        ## Train the model from the beginning
        #####################################
        completed_epochs = []
        # Store running averages of train/test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []
        # Make epochs 1-indexed for better prints
        epoch_range = range(1, N_EPOCHS + 1)
        # Train and test for each epoch
        for epoch in epoch_range:
            train(epoch)
            test(epoch)
            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
            # keep track of stats for each epoch
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            completed_epochs.append(epoch)
            # reset losses before next epoch
            train_losses = []
            test_losses = []
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        evaluate_performance(completed_epochs, avg_train_losses, avg_test_losses)

    ############
    ## The End
    ############
    print("\nEnd: {}".format(datetime.now()))
