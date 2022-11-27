import os
import csv
import pickle
import numpy as np
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

# Setup paths for saving model + optimizer
MODEL_PATH = "results/model.pth"
OPTIMIZER_PATH = "results/optimizer.pth"
# Setup paths for accessing data
TRAIN_CSV = "train/train.csv"
TRAIN_ROOT = "train/images/"
TEST_CSV = "val/val.csv"
TEST_ROOT = "val/images/"


def Net():
    """
    Retrieve the pre-constructed CNN model
    """
    # img_channel == 3 because RGB
    # num_classes == 7 because Pose(q,r)
    return ResNet50(img_channel=3, num_classes=7)


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
        csv_file=TEST_CSV,
        root_dir=TEST_ROOT,
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
    # Save the current state of the Model and the Optimizer
    # so we can load the latest state later on
    torch.save(net.state_dict(), MODEL_PATH)
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)


def load_model():
    print("Loading the saved model: `{}`".format(MODEL_PATH))
    net = Net()
    net.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded.")
    return net


def evaluate_performance(train_counter, train_losses, test_counter, test_losses):
    # print("train_counter", train_counter)
    # print("train_losses", train_losses)
    # print("test_counter", test_counter)
    # print("test_losses", test_losses)
    output = "figures/loss.png"
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Mean Square Error (MSE) Loss")
    plt.savefig(output)
    print("Performance evaluation saved to: `{}`".format(output))


# def write_output_csv(predictions, metadata):
#     """
#     Write model predictions to output submission CSV
#     """
#     print("Write the predicted output to CSV...")
#     print("\t predictions length: {}".format(len(predictions)))
#     print("\t metadata length: {}".format(len(metadata)))
#     with open("predictions_submission.csv", "w", newline="") as csvfile:
#         fieldnames = ["filename", "sequence", "Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for i in range(0, len(predictions)):
#             row = {
#                 "filename": metadata[i]["filename"],
#                 "sequence": metadata[i]["sequence"],
#                 "Tx": predictions[i][0],
#                 "Ty": predictions[i][1],
#                 "Tz": predictions[i][2],
#                 "Qx": predictions[i][3],
#                 "Qy": predictions[i][4],
#                 "Qz": predictions[i][5],
#                 "Qw": predictions[i][6],
#             }
#             writer.writerow(row)


if __name__ == "__main__":
    #############################################
    ## Print start time to keep track of runtime
    #############################################
    print("Start: {}".format(datetime.now()))
    ###########################
    ## Initialize the CNN model
    ###########################
    net = Net()
    # specify the optimizer hyperparameters
    learning_rate = 0.01
    momentum = 0.5
    # specify the optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
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
    BATCH_SIZE = 1
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
    n_epochs = 1  # 3
    log_interval = 10
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
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            inputs = batch["image"].float()
            labels = batch["pose"].float()  # .reshape(1, 7 * batch_size_train)
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
            if i % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
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

    ################################
    ################################
    ### Test the Whole Test Dataset
    ################################
    ################################
    def test():
        # Let us look at how the network performs on the whole dataset.
        test_loss = 0
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # for data in test_loader:
            for i, batch in enumerate(test_loader, 0):
                inputs = batch["image"].float()
                labels = batch["pose"].float()
                # labels = batch["pose"].float().reshape(1, 7)
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # check the loss
                test_loss = criterion(outputs, labels)
                ######################
                #### Scoring System:
                ######################
                # print("outputs:", outputs)
                # print("labels:", labels)
                # print(outputs == labels)                  # >>> tensor([[False, False, False, False, False, False, False]])
                # print((outputs == labels).sum())          # >>>  tensor(0)
                # print((outputs == labels).sum().item())   # >>>  0
                correct += (outputs == labels).sum().item()
                # print and store statistics
                if i % log_interval == 0:
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

    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    # Make epochs 1-indexed for better prints
    epoch_range = range(1, n_epochs + 1)
    # Iterate through each epoch
    # doing the train/test steps
    for epoch in epoch_range:
        print("Start Training...")
        train(epoch)
        print("Finished Training.")
        print("Start Testing...")
        test()
        print("Finished Testing.")
    ###################################
    ## Evaluate the model's performance
    ###################################
    evaluate_performance(train_counter, train_losses, test_counter, test_losses)

    ############
    ## The End
    ############
    print("End: {}".format(datetime.now()))
