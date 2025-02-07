# Mars-Object-Classifier

## This program will look at images of different objects on Mars and be able to detect what object is in the image using a Tensorflow CNN model.
## After model is made and predicts, graphs accuracy changes using Matplotlib

# model.py

## Makes the model for object classificiation

# preprocessing.py

## Preprocesses the images into arrays using Numpy. Each array includes multiple RGB values which are connected to other images of similar labels.

# read_data.py

## Reads data into Pandas DataFrame using loops to make sure each label is cleaned and corresponds to a different image. Proof of classification work is plotted on preprocessing.py

# plot.py 

## Plots the model accuracy changes using Matplotlib.
