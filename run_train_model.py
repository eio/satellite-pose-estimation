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

# Setup tunable constants
N_EPOCHS = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
# Setup path for saving model + optimizer
SAVED_MODEL_PATH = "results/model+optimizer.pth"
# Setup paths for accessing data
TRAIN_CSV = "train/train.csv"
TRAIN_ROOT = "train/images/"
VALIDATION_CSV = "val/val.csv"
VALIDATION_ROOT = "val/images/"
# Setup path for output predictions
PREDICTIONS_OUTPUT_CSV = "predictions_submission.csv"


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


def evaluate_performance(train_counter, train_losses, test_counter, test_losses):
    # print("train_counter", train_counter)
    # print("train_losses", train_losses)
    # print("test_counter", test_counter)
    # print("test_losses", test_losses)
    FIGURE_OUTPUT = "figures/loss.png"
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Mean Square Error (MSE) Loss")
    plt.savefig(FIGURE_OUTPUT)
    print("Performance evaluation saved to: `{}`".format(FIGURE_OUTPUT))


def write_output_csv(predictions):
    """
    Write model predictions to output submission CSV
    """
    metadata = pd.read_csv(VALIDATION_CSV)
    print("Write the predicted output to: {}...".format(PREDICTIONS_OUTPUT_CSV))
    print("\t predictions length: {}".format(len(predictions)))
    print("\t metadata length: {}".format(len(metadata)))
    with open(PREDICTIONS_OUTPUT_CSV, "w", newline="") as csvfile:
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
    net = Net()
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
            inputs = batch["image"].float()
            labels = batch["pose"].float()
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
        print("\nStart Testing...")
        # Initialize array to store all predictions
        predictions = []
        # Let us look at how the network performs on the whole dataset.
        test_loss = 0
        correct = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # for data in test_loader:
            for i, batch in enumerate(test_loader, 0):
                inputs = batch["image"].float()
                labels = batch["pose"].float()
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
                ######################
                #### Scoring System:
                ######################
                # print(outputs == labels)                  # >>> tensor([[False, False, False, False, False, False, False]])
                # print((outputs == labels).sum())          # >>>  tensor(0)
                # print((outputs == labels).sum().item())   # >>>  0
                correct += (outputs == labels).sum().item()
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
        print("Finished Testing.")
        # Write the predicted poses to an output CSV
        # in the submission format expected
        write_output_csv(predictions)

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
        # Make epochs 1-indexed for better prints
        epoch_range = range(1, N_EPOCHS + 1)
        # Iterate through each epoch
        # doing the train/test steps
        for epoch in epoch_range:
            train(epoch)
            test(epoch)
        ###################################
        ## Evaluate the model's performance
        ###################################
        evaluate_performance(train_counter, train_losses, test_counter, test_losses)

    ############
    ## The End
    ############
    print("End: {}".format(datetime.now()))
