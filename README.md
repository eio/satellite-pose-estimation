# Installing packages

Required Python packages are listed in `requirements.txt`

# The code

`run_train_model.py` is the code that trains the CNN on provided satellite pose images and labels

`SatellitePoseDataset.py` defines the custom PyTorch dataset that is used in `run_train_model.py`

`test_SatellitePoseDataset.py` tests that `SatellitePoseDataset.py` is working as expected, and uses functions in `visually_test_pytorch_setup.py` which itself uses functions in `visualize_data.py`


# Resources

PyTorch complete example - CNN with MNIST dataset:

	https://nextjournal.com/gkoehler/pytorch-mnist

Creating and validating a custom PyTorch Dataloader:

	https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

	https://glassboxmedicine.com/2022/01/21/building-custom-image-data-sets-in-pytorch-tutorial-with-code/
