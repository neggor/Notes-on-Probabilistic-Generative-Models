from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class f3_class(tfb.Bijector):

    
    def __init__(self, a, validate_args=False):
        super().__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1,
        name='f3_class',
        is_constant_jacobian=True
        )
        self.a = tf.cast(a, tf.float32)
        self.event_ndim = 1

    def _forward(self, x):
        batch_ndim = len(x.shape) - self.event_ndim
        x = tf.cast(x, tf.float32)
        x0 = tf.expand_dims(x[..., 0], batch_ndim)
        x1 = tf.expand_dims(x[..., 1], batch_ndim)
        y0 = x0
        y1 = x1 + ((x0**2) * self.a)
        return tf.concat((y0, y1), axis=-1)
    
    def _inverse(self, y):
        batch_ndim = len(y.shape) - self.event_ndim
        y = tf.cast(y, tf.float32)
        y0 = tf.expand_dims(y[..., 0], batch_ndim)
        y1 = tf.expand_dims(y[..., 1], batch_ndim)
        x0 = y0
        x1 = y1 - ((x0**2) * self.a)
        return tf.concat((x0, x1), axis=-1)
    
    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

class f4_class(tfb.Bijector):
    def __init__(self, theta, validate_args=False):
        super().__init__(
          validate_args=validate_args,
          forward_min_event_ndims=1,
          inverse_min_event_ndims=1,
          name='f4_class',
          is_constant_jacobian=True
        )

        self.cos_theta = tf.math.cos(theta)
        self.sin_theta = tf.math.sin(theta)
        self.event_ndim = 1

    def _forward(self, x):
        batch_ndim = len(x.shape) - self.event_ndim
        x0 = tf.expand_dims(x[..., 0], batch_ndim)
        x1 = tf.expand_dims(x[..., 1], batch_ndim)
        y0 = self.cos_theta * x0 - self.sin_theta * x1
        y1 = self.sin_theta * x0 + self.cos_theta * x1
        return tf.concat((y0, y1), axis=-1)

    def _inverse(self, y):
        batch_ndim = len(y.shape) - self.event_ndim
        y0 = tf.expand_dims(y[..., 0], batch_ndim)
        y1 = tf.expand_dims(y[..., 1], batch_ndim)
        x0 = self.cos_theta * y0 + self.sin_theta * y1
        x1 = -self.sin_theta * y0 + self.cos_theta * y1
        return tf.concat((x0, x1), axis=-1)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

def generate_distribution(a, theta):
    '''
    Generates a transformed distribution
    given a and theta.

    - a: Shift strength.
    - theta: Rotation strength (radians)
    '''
    #Bijectors
    f1 = tfb.Shift((0., -2.))
    f2 = tfb.Scale((1., 0.5))
    f3 = f3_class(a)
    f4 = f4_class(theta)
    f5 = tfp.bijectors.Tanh()
    transformation = tfb.Chain([f5, f4, f3, f2, f1])
    # Distribution
    distribution = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag( loc = [[0, 0]],
                                    scale_diag= [[0.3, 0.3]]),
                                     bijector = transformation)
    return distribution

def get_densities(transformed_distribution):
    '''
    Returns heatmap values.
    '''
    # Points to evaluate:
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    inputs = np.transpose(np.stack((X, Y)), [1, 2, 0])

    # Evaluate density:
    batch_shape = transformed_distribution.batch_shape
    Z = transformed_distribution.prob(np.expand_dims(inputs, 2)).numpy()
    Z = np.transpose(Z, list(range(2, 2+len(batch_shape))) + [0, 1])

    # Correct numerical problems:
    Z[np.isnan(Z)] = 0

    return Z.squeeze()

def get_image_array_from_density_values(Z, heatmap, size= 0.36):
    """
    This function takes a numpy array Z of density values of shape (100, 100)
    and returns an integer numpy array of shape (36, 36, 3) of pixel values for an image.
    """
    assert Z.shape == (100, 100)
    fig = Figure(figsize=(size, size))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))


    ax.contourf(X, Y, Z, cmap= heatmap, levels=100)
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_from_plot

def generate_data(n_train, n_val, n_test):

    heatmap = ['inferno', 'hot', 'nipy_spectral']

    my_images = []

    print('Generating data...')
    for i in tqdm(range(n_train + n_val + n_test)):
        heatmap_index = np.random.randint(0, 3)
        a = tf.random.normal(shape = (1, ),
                            mean = np.random.normal(3, 0.7, size = 1),
                            stddev= np.random.lognormal(1, 0.5, size = 1))
        theta = tf.random.uniform(shape = (1, ), minval=0, maxval= 2*np.pi)
        my_distri = generate_distribution(a, theta)
        my_images.append(get_image_array_from_density_values(get_densities(my_distri),
                                                    heatmap = heatmap[heatmap_index]))

    my_images = np.array(my_images)

    np.save('Data/weird_dataset_train.npy', my_images[:n_train])
    np.save('Data/weird_dataset_val.npy', my_images[n_train:n_train + n_val])
    np.save('Data/weird_dataset_test.npy', my_images[n_train + n_val:])


def main():
    generate_data(n_train = 10**3, n_val = 10**2, n_test = 10**2)


if __name__ == "__main__":
    main()

