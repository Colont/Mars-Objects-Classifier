from read_data import read_data
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

cwd = os.getcwd()
vocab_df, train_df, test_df, val_df = read_data()

# Identify Labels based on the terms

# Add normalized image values to a matrix
def normalization(train_df, test_df, vocab_df, val_df):
    '''
    Normalize the pixel values of the images into a matrix of values
    Param: Train_df (training dataset)
    Param: Test_df (testing dataset)
    Param: Vocab_df (vocab dataset)
    '''
    # Map the labels from each df to the vocab name in the vocab df
    train_labels = train_df['ID'].map(vocab_df.set_index('ID')['Term'])
    test_labels = test_df['ID'].map(vocab_df.set_index('ID')['Term'])
    validation_labels = val_df['ID'].map(vocab_df.set_index('ID')['Term'])

    train_normalized = []
    test_normalized = []
    validation_normalized = []
    # Looks at the column in train_df of images and converts it into a pixel array
    for url in train_df['Picture']:
        img = load_img(cwd + '/msl-images/' + url)
        img_array = img_to_array(img)
    
        # Normalize pixel values
        img_array_normalized = img_array / 255.0
    
        # Append normalized array to the list
        train_normalized.append(img_array_normalized)
    for url in test_df['Picture']:
        img = load_img(cwd + '/msl-images/' + url)
        img_array = img_to_array(img)
    
        img_array_normalized = img_array / 255.0
    
        test_normalized.append(img_array_normalized)
    for url in test_df['Picture']:
        img = load_img(cwd + '/msl-images/' + url)
        img_array = img_to_array(img)
    
        img_array_normalized = img_array / 255.0
    
        test_normalized.append(img_array_normalized)
        # Display a few of the training images
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_normalized[i])
            plt.xlabel(train_labels[i])
        plt.show()
    return train_normalized, test_normalized, validation_normalized

def preprocess():
    train_norm, test_norm, validation_norm = normalization(train_df, test_df, vocab_df, val_df)

if __name__ == '__main__':
    preprocess()