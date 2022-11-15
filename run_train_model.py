# import pickle
import os
import csv
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        in_channels = 3  # RGB
        out_channels = 7  # pose array
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        self.conv2_drop = nn.Dropout2d()
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_channels)

    def forward(self, x):
        conv1x = self.conv1(x)
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        x = F.relu(F.max_pool2d(conv1x, 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # specify some required weights
        weight = torch.Tensor([[1, 1, 1, 1, 1, 1, 1]])
        print("`forward` function:")
        print("\t`x.shape`: {}".format(x.shape))
        print("\t`weight.shape`:{} \n\n".format(weight.shape))
        # https://pytorch.org/docs/stable/nn.functional.html
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear
        return F.linear(x, weight)


def build_model(learning_rate, momentum):
    print("Build the model...")
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer


def build_data_loaders(batch_size_train, batch_size_test, img_downscale_size):
    # Create the Train dataset
    train_dataset = SPD.SatellitePoseDataset(
        csv_file="train/train.csv",
        root_dir="train/images/",
        transform=torchvision.transforms.Compose(
            [SPD.Rescale(img_downscale_size), SPD.ToTensor()]
        ),
    )
    # Create the Test dataset
    test_dataset = SPD.SatellitePoseDataset(
        csv_file="val/val.csv",
        root_dir="val/images/",
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


def evaluate_performance(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.savefig("figures/loss.png")


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
    learning_rate = 0.01
    momentum = 0.5
    network, optimizer = build_model(learning_rate, momentum)
    ##################
    n_epochs = 3
    log_interval = 10
    # cuDNN uses nondeterministic algorithms which are disabled here
    torch.backends.cudnn.enabled = False
    # For repeatable experiments we have to set random seeds
    # for anything using random number generation
    random_seed = 1
    torch.manual_seed(random_seed)
    #################################################################
    ## Load the custom SatellitePoseDataset into PyTorch DataLoaders
    #################################################################
    # TODO: change batch size
    # batch_size_train = 3480
    # batch_size_test = 3480
    batch_size_train = 10
    batch_size_test = 10
    # downscale by a factor of 4 from original size: (1440,1880)
    img_downscale_size = (270, 360)
    train_loader, test_loader = build_data_loaders(
        batch_size_train, batch_size_test, img_downscale_size
    )
    #########################
    ## Initialize the output
    #########################
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, batch in enumerate(train_loader):
            # ensure the same type by calling float()
            data = batch["image"].float()
            target = batch["pose"].float()
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), "results/model.pth")
                torch.save(optimizer.state_dict(), "results/optimizer.pth")

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            print("Iterating through `test_loader` contents...")
            for batch in test_loader:
                # ensure the same type by calling float()
                data = batch["image"].float()
                target = batch["pose"].float()
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    ###################
    ## Train the model
    ###################
    print("Initial call to `test()`...")
    test()
    # # Train through the specified epochs
    # print("Entering epoch training loop...")
    # for epoch in range(1, n_epochs + 1):
    #     train(epoch)
    #     test()
    # ###################################
    # ## Evaluate the model's performance
    # ###################################
    # evaluate_performance(train_counter, train_losses, test_counter, test_losses)

    # ############
    # ## The End
    # ############
    # print("End: {}".format(datetime.now()))
