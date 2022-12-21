# Description

This project adapts a ResNet50 model architecture to perform pose estimation on several series of satellite images (both real and synthetic).

For more information, please see the SPARK Challenge ( https://cvi2.uni.lu/spark2022/ ) organized as part of the AI4Space workshop, in conjunction with the European Conference on Computer Vision (ECCV 2022).


# Installing packages

See `requirements.txt` and make sure to also install `cudatoolkit` if you plan to run with a GPU.


# Running the code

To train the model:

	python run_train_model.py

To train the model and output all print statements to a local file:

	python run_train_model.py > LOGS.txt

To load and test the saved, pre-trained model on the `test_real` dataset:

	python run_train_model.py -l -tr

To load and test the saved, pre-trained model on the `test_synthetic` dataset:

	python run_train_model.py -l -ts


# Output and results

After training, the trained model and optimizer are stored in `results/model+optimizer.pth`

Model predictions are stored as CSV files in `predictions/`

Loss function plots are stored in `figures/`


# Resources

PyTorch Resnet implementation taken from:

	https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py

Helpful tutorials:

	https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
	
	https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Creating and validating a custom PyTorch Dataloader:

	https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

	https://glassboxmedicine.com/2022/01/21/building-custom-image-data-sets-in-pytorch-tutorial-with-code/

Reference: PyTorch complete example - CNN with MNIST dataset:

	https://nextjournal.com/gkoehler/pytorch-mnist

Loss Functions:

	https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
	https://neptune.ai/blog/pytorch-loss-functions#:~:text=Broadly%20speaking
