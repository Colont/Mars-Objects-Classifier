import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from preprocessing import preprocess
from read_data import read_data

train_norm, test_norm = preprocess()
vocab_df, train_df, test_df, val_df = read_data()

