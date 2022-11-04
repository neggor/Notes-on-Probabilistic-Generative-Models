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

from datetime import datetime
import matplotlib.animation as anim
from IPython import display
from VAE import make_model

plt.style.use('dark_background')

LATENT_DIM = 50


def load_model_and_weights():
    '''
    Returns trained decoder
    '''
    encoder, decoder, vae = make_model(return_pieces = True)
    vae.load_weights('Model/Vae_weights.h5')

    return decoder


def interpolation(latent_size, decoder, interpolation_length=500):

    fig = plt.figure(figsize=(10, 10))  
    ax = fig.add_subplot(1,1,1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axis('off')
    img = ax.imshow(np.zeros((36, 36, 3)))

    freqs = np.random.uniform(low=0.1, high=0.2, size=(latent_size,))
    phases = np.random.randn(latent_size)
    input_points = np.arange(interpolation_length)
    latent_coords = []

    for i in range(latent_size):
        latent_coords.append(
            2 * np.sin((freqs[i]*input_points + phases[i])
            ).astype(np.float32))

    def animate(i): 
        z = tf.constant([coord[i] for coord in latent_coords])
        img_out = np.squeeze(decoder(z[np.newaxis, ...]).mean().numpy())
        img.set_data(np.clip(img_out, 0, 1))
        return (img,)

    return anim.FuncAnimation(fig, animate, frames=interpolation_length, 
                              repeat=False, blit=True, interval=150)

def main():
    decoder = load_model_and_weights()
    my_interpolation = interpolation(LATENT_DIM, decoder, interpolation_length=500)
    writervideo = anim.FFMpegWriter(fps=10)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print('Generating video, please wait...')
    my_interpolation.save(f'Outputs/interpoplation_{current_time}.mp4',
                            writer = writervideo)
    print('Done!')

    #display.HTML(my_interpolation.to_html5_video())

plt.close()
if __name__ == "__main__":
    main()