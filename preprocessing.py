from read_data import main
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os

cwd = os.getcwd()

vocab_df, train_df, test_df, val_df = main()

# Identify Labels based on the terms
train_df['Labels'] = train_df['ID'].map(vocab_df.set_index('ID')['Term'])
test_df['Labels'] = test_df['ID'].map(vocab_df.set_index('ID')['Term'])

# Add normalized image values to a matrix
normalized_images = []
for url in train_df['Picture']:
    img = load_img(cwd + '/msl-images/' + url)
    img_array = img_to_array(img)
    
    # Normalize pixel values
    img_array_normalized = img_array / 255.0
    
    # Append normalized array to the list
    normalized_images.append(img_array_normalized)
print(normalized_images)