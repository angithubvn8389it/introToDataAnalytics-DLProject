import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_and_train_autoencoder(scaled_features, encoding_dim=10, epochs=30, batch_size=32):
    print("\n--- Training Autoencoder for Dimensionality Reduction ---")
    input_dim = scaled_features.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    
    # Encoder model to extract latent features
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(
        scaled_features, scaled_features,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )
    
    latent_features = encoder.predict(scaled_features)
    print(f"Autoencoder extraction complete. Latent shape: {latent_features.shape}")
    
    return autoencoder, encoder, latent_features
