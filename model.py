import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from read_data import read_data
from preprocessing import preprocess


(X_train, y_train), (X_test, x_val), (X_val, y_val) = preprocess()
vocab_df, train_df, test_df, val_df = read_data()
num_classes = vocab_df['ID'].nunique()
def build_model():
    
    model = tf.keras.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),  
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  
                  metrics=['accuracy'])

    return model

# Train model
def train_model():
    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32  
    )
    return history, model

if __name__ == '__main__':
    hist, trained_model = train_model()