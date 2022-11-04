import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import os
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D, 
                                     UpSampling2D, BatchNormalization)
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.animation as anim
from IPython.display import HTML

plt.style.use('dark_background')
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
LATENT_DIM = 50

def data_loader():

    train_data = np.load("Data/weird_dataset.npy")
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.map(lambda x: x/255)
    train_data = train_data.map(lambda x: (x, x))

    train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    val_data = np.load("/Data/weird_dataset_val.npy")
    val_data = tf.data.Dataset.from_tensor_slices(val_data)
    val_data = val_data.map(lambda x: x/255)
    val_data = val_data.map(lambda x: (x, x))

    val_data = val_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    test_data = np.load("/Data/weird_dataset_test.npy")
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    test_data = test_data.map(lambda x: x/255)
    test_data = test_data.map(lambda x: (x, x))

    test_data = test_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return train_data, val_data, test_data

def get_encoder(latent_dim, kl_regularizer):

    model = Sequential(
        [Conv2D(36, 4, 2, 'SAME',
               input_shape = (36, 36, 3),
               activation = 'elu',
               name = 'e1'),
         BatchNormalization(),
         Conv2D(36, 4, 2, 'SAME',
                activation = 'elu', name = 'e2'),
         BatchNormalization(),
         Conv2D(72, 4, 2, 'SAME',
                activation = 'elu', name = 'e3'),
         BatchNormalization(),
         Conv2D(144, 4, 2, 'SAME',
                activation = 'elu', name = 'e4'),
         BatchNormalization(),
         Flatten(),
         Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)),
         tfpl.MultivariateNormalTriL(latent_dim,
                                    activity_regularizer = kl_regularizer)
                       ])
                        
    return model

def get_decoder(latent_dim):

    model = Sequential(
        [Dense(4050, input_shape = (latent_dim, ), activation = 'elu', name = 'd1'),
         Reshape((9, 9, 50)),
         #UpSampling2D(2),
         Conv2D(72, 3, activation = 'elu', padding = 'SAME', name = 'd2'),
         UpSampling2D(2),
         Conv2D(36, 3, activation = 'elu', padding = 'SAME', name = 'd3'),
         #UpSampling2D(2),
         Conv2D(36, 3, activation = 'elu', padding = 'SAME', name = 'd4'),
         UpSampling2D(2),
         Conv2D(72, 3, activation = 'elu', padding = 'SAME', name = 'd5'),
         Conv2D(3, 3, activation = 'linear', padding = 'SAME', name = 'd6'),
         Flatten(),
         tfpl.IndependentBernoulli(event_shape = (36, 36, 3))
        ])
                        
    return model

def get_prior(latent_dim):

    prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros([latent_dim]),
                scale_diag=tf.ones([latent_dim])
                )

    return prior

def get_kl_regularizer(prior_distribution):
    reg = tfpl.KLDivergenceRegularizer(
        prior_distribution,
        test_points_fn = lambda q: q.sample(10)) # MC approximation 10
                                                 # samples
    return reg

def reconstruction_loss(batch_of_images, decoding_dist):
    # This works because the decoder outputs
    # a probability distribution!
    return -tf.math.reduce_mean(decoding_dist.log_prob(batch_of_images))

def make_model(return_pieces = False):
    '''
    Returns Variational Autoencoder.
    '''
    # Prior and KL loss:
    prior = get_prior(LATENT_DIM)
    regularizer = get_kl_regularizer(prior)

    # Architecture:
    encoder = get_encoder(LATENT_DIM, regularizer)
    decoder = get_decoder(LATENT_DIM)

    # Construct the model
    vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    if return_pieces:
        return encoder, decoder, vae
    else: 
        return vae

def main():

    vae = make_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Data
    train_data, val_data, test_data = data_loader()

    # Training stuff:
    vae.compile(optimizer=optimizer, loss=reconstruction_loss)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights = True)
        

    history = vae.fit(  train_data,
                        validation_data=val_data,
                        epochs=100,
                        callbacks=[callback])

    vae.save_weights('Model/Vae_weights.h5')


    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.legend()
    
    plt.show(block=False)
    plt.pause(5)
    plt.close("all")

if __name__ == "__main__":
    main()
