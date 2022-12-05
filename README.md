# Installing packages

	pip3 install torch torchvision 'pillow==9.2.0'
	pip3 install matplotlib
	pip3 install pandas
	pip3 install opencv-python
	pip3 install scipy
	pip3 install scikit-image

# Running the code

To train the model from scratch:

	python run_train_model.py

To train the model from scratch and output all print statements to a local file:

	python run_train_model.py > LOGS.txt

To load and test the saved, pre-trained model:

	python run_train_model.py -l

...or:

	python run_train_model.py --load

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
