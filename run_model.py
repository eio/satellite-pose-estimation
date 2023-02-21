import argparse
from numpy import mean
from datetime import datetime

# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim

# Local scripts
from model_architectures.pytorch_resnet import ResNet50
from save_and_load import save_model, load_model, plot_loss, write_output_csv
import SatellitePoseDataset as SPD
from SatelliteDataLoaders import (
    build_data_loaders,
    build_final_test_data_loader,
    VALIDATION_CSV,
)

# Check for CUDA / GPU Support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running with device: {}".format(DEVICE))
# Setup tunable constants
N_EPOCHS = 10
BATCH_SIZE = 1
LOG_INTERVAL = 50
# Specify the loss function
# Mean Square Error (MSE) is the most commonly used regression loss function.
# MSE is the sum of squared distances between our target variable and predicted values.
# https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
criterion = nn.MSELoss()

# Ensure deterministic behavior:
# cuDNN uses nondeterministic algorithms which are disabled here
torch.backends.cudnn.enabled = False
# For repeatable experiments we have to set random seeds
# for anything using random number generation
random_seed = 1111
torch.manual_seed(random_seed)


def Model():
    """
    Retrieve the pre-constructed CNN model
    """
    # img_channel == 3 because [R, G, B]
    # num_classes == 7 because ["Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
    return ResNet50(img_channel=3, num_classes=7)


def Optimizer(model):
    """
    Create SGD optimizer with specified hyperparameters
    """
    learning_rate = 0.01
    momentum = 0.5
    # TODO: try Adam
    # optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def train_process(optimizer, model, criterion, train_loader, epoch):
    """
    Train the model
    """
    print("\nStart Training for Epoch #{}...".format(epoch))
    # Initialize losses
    train_losses = []
    for i, batch in enumerate(train_loader, 0):
        # Convert inputs/labels (aka data/targets)
        # to float values, and send Tensors to GPU device (if CUDA-compatible)
        inputs = batch["image"].float().to(DEVICE)
        labels = batch["pose"].float().to(DEVICE)
        # labels = batch["pose"].float().reshape(1, 7 * batch_size_train)
        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward propagation
        outputs = model(inputs)
        # Calculate the loss
        loss = criterion(outputs, labels)
        loss.backward()
        # Store the loss value for this batch
        train_losses.append(loss.item())
        # Perform an optimization step (parameter update)
        optimizer.step()
        # Print some statistics based on the log interval
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
    print("Finished Training for Epoch #{}.".format(epoch))
    #####################
    ## Save the Model  ##
    #####################
    save_model(epoch, model, optimizer)
    # Return the train losses from this epoch
    return train_losses


def test_process(model, criterion, test_loader, epoch, sample_csv=VALIDATION_CSV):
    """
    Test the model
    """
    print("\nStart Testing for Epoch {}...".format(epoch))
    # Initialize losses
    test_losses = []
    # Initialize array to store all predictions
    predictions = []
    # Initialize correct prediction count
    correct = 0
    # Since we're not training,
    # we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # for data in test_loader:
        for i, batch in enumerate(test_loader, 0):
            # Convert inputs/labels (aka data/targets)
            # to float values, and send Tensors to GPU device (if CUDA-compatible)
            inputs = batch["image"].float().to(DEVICE)
            labels = batch["pose"].float().to(DEVICE)
            # labels = batch["pose"].float().reshape(1, 7)
            # calculate outputs by running images through the model
            outputs = model(inputs)
            # Store the predicted outputs
            prediction = outputs.cpu().numpy().flatten()
            predictions.append(prediction)
            # Calculate the loss
            test_loss = criterion(outputs, labels)
            # Store the loss value for this batch
            test_losses.append(test_loss.item())
            ## Consider prediction to be correct
            ## if `test_loss` is "close enough" to a perfect score of 0.0
            close_enough = 0.001
            if test_loss <= close_enough:
                correct += 1
            # Print some statistics based on the log interval
            if i % LOG_INTERVAL == 0:
                print(
                    "Test: [{}/{} ({}%)]\tLoss: {}".format(
                        i,  # i * len(batch),
                        len(test_loader.dataset),
                        100.0 * i / len(test_loader),
                        test_loss.item(),
                    )
                )
    print(
        "Test set: Avg. loss: {}, Accuracy: {}/{} ({}%)".format(
            mean(test_losses),
            correct,
            len(test_loader.dataset),
            100.0 * (correct / len(test_loader.dataset)),
        )
    )
    print("Finished Testing for Epoch {}.".format(epoch))
    # Write the predicted poses to an output CSV
    # in the submission format expected
    test_dataset_name = test_loader.dataset.root_dir.split("/")[1]
    write_output_csv(predictions, sample_csv, epoch, test_dataset_name)
    # Return the test losses from this epoch
    return test_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags specifying if script should run in "load" or "train" mode
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-ts", "--synthetic", action="store_true")
    parser.add_argument("-tr", "--real", action="store_true")
    args = parser.parse_args()
    LOAD_MODEL = args.load
    TEST_SYNTHETIC = args.synthetic
    TEST_REAL = args.real
    #############################################
    ## Print start time to keep track of runtime
    #############################################
    print("Start: {}".format(datetime.now()))
    ###########################
    ## Initialize the CNN model
    ###########################
    # Send model to GPU device (if CUDA-compatible)
    model = Model().to(DEVICE)
    optimizer = Optimizer(model)
    # Configure batch and downscale sizes
    batch_size_train = BATCH_SIZE
    batch_size_test = BATCH_SIZE
    # Downscale by a factor of 4 from original size: (1440,1080)
    IMG_WIDTH = 1440 / 4  # 360
    IMG_HEIGHT = 1080 / 4  # 270
    img_downscale_size = (IMG_HEIGHT, IMG_WIDTH)
    #########################################
    ## Initialize losses for loss plot output
    #########################################
    train_losses = []
    test_losses = []
    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    if LOAD_MODEL == True:
        ###############################################
        # Load the previously saved model and optimizer
        ###############################################
        model, optimizer = load_model()
        # Update test dataset, overwriting `test_loader` variable
        if TEST_SYNTHETIC == True:
            print("Testing for images in: {}".format(TEST_SYNTHETIC_ROOT))
            test_loader = build_final_test_data_loader(
                batch_size_test,
                img_downscale_size,
                TEST_SYNTHETIC_CSV,
                TEST_SYNTHETIC_ROOT,
            )
            # Test the loaded model on the synthetic data
            test_losses = test_process(
                model, criterion, test_loader, 1, TEST_SYNTHETIC_CSV
            )
        elif TEST_REAL == True:
            print("Testing for images in: {}".format(TEST_REAL_ROOT))
            test_loader = build_final_test_data_loader(
                batch_size_test, img_downscale_size, TEST_REAL_CSV, TEST_REAL_ROOT
            )
            # Test the loaded model on the real data
            test_losses = test_process(model, criterion, test_loader, 1, TEST_REAL_CSV)
        else:
            print(
                "FAIL: Please specify whether to test the Real or Synthetic dataset (-tr or -ts)"
            )
            sys.exit()
    else:
        #################################################################
        ## Load the custom SatellitePoseDataset into PyTorch DataLoaders
        #################################################################
        train_loader, test_loader = build_data_loaders(
            batch_size_train, batch_size_test, img_downscale_size
        )
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
            # Run the training process
            train_losses = train_process(
                optimizer, model, criterion, train_loader, epoch
            )
            # Run the testing process
            test_losses = test_process(
                model, criterion, test_loader, epoch, VALIDATION_CSV
            )
            train_loss = mean(train_losses)
            test_loss = mean(test_losses)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
            # Keep track of stats for each epoch
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            completed_epochs.append(epoch)
            # Reset losses before next epoch
            train_losses = []
            test_losses = []
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        plot_loss(completed_epochs, avg_train_losses, avg_test_losses)

    ############
    ## The End
    ############
    print("\nEnd: {}".format(datetime.now()))
