# CustomANN

This repo contains the code for the Turion Space Machine Learning Integration Intern Take Home Assessment materials.

There are two datasets that been investigated: the generated dataset CIFAR-10 and real-world dataset MS COCO

#Getting Start:

Assume a new conda environment, perform this command:

`pip install -r requirements.txt`

#Downloading the dataset:

CIFAR-10: using the CIFAR-10 API within the Jupyter File (automatically downloading)

COCO-MS: [link](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) 

# COCO Dataset

The `CocoDataset2.py` contain a dataset class constructed for MS COCO to be compiled with PyTorch DataLoader

# Model Training and Uncertainty Estimation Code:

The model training and visualization are presented within each Jupyter file. where the outer loop contain the epochs that run (can alternate for the desired epochs)

For uncertainty quantification, a technique of Y. Gal "What my deep model doesn't know..." [link](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html) is being replicated.

# Future Work 

Only successfully trained and illustrate the uncertainty for generated dataset, CIFAR-10

MS COCO posed a difficulty in matching the size and required a stronger memory training devices.

