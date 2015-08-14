# Credits to Theano Tutorial
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MLP(object):
    def __init__(self, np_rng, theano_rng=None, input=None,
                 layers=None):

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(theano.shared(
                value=np.asarray(np_rng.uniform(
                    low=-4 * np.sqrt(6. / (layers[i] + layers[i + 1])),
                    high=4 * np.sqrt(6. / (layers[i] + layers[i + 1])),
                    size=(layers[i], layers[i + 1])),
                    dtype=theano.config.floatX),
                name='W_' + str(i) + '_to_' + str(i + 1),
                borrow=True))
            biases.append(theano.shared(
                value=np.zeros(
                    layers[i + 1],
                    dtype=theano.config.floatX),
                name='b_' + str(i + 1),
                borrow=True))

        self.layers = layers
        self.num_layers = len(layers)
        self.weights = weights
        self.biases = biases
        self.theano_rng = theano_rng
        self.x = input
        self.params = self.weights + self.biases

    def get_corrupted_input(self, input, corruption_level):
        # an array of 0s and 1s where
        # p(1) = 1 - corruption_level
        # p(0) = corruption_level
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_layer_from_previous_layer(self, previous_layer, layer_number):
        return T.nnet.sigmoid(
            T.dot(previous_layer, self.weights[layer_number - 1]) +
            self.biases[layer_number - 1])

    def get_layer_from_input(self, input, layer_number):
        layer = input
        for i in range(layer_number):
            layer = self.get_layer_from_previous_layer(layer, i + 1)
        return layer

    def get_output_from_layer(self, layer, layer_number):
        layer_ = layer
        for i in np.arange(layer_number + 1, self.num_layers):
            layer_ = self.get_layer_from_previous_layer(layer_, i)
        return layer_

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        z = self.get_output_from_layer(tilde_x, 0)
        # loss = - T.sum(self.x * T.log(z) +
        #                (1 - self.x) * T.log(1 - z), axis=1)
        # cost = T.mean(loss)
        cost = T.mean((self.x - z) ** 2)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.params, gparams)]
        return (cost, updates)


def build_model(layers=[784, 500, 784]):
    # Basic settings
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # Define model
    mlp = MLP(np_rng=rng, theano_rng=theano_rng, input=x,
              layers=layers)

    return index, x, mlp


def train(index, x, mlp, corruption_level=0., learning_rate=0.1,
          dataset='mnist.pkl.gz', batch_size=100, epochs=15):
    # get cost and updates
    cost, updates = mlp.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate)

    # Create training function
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    train_func = theano.function(
        [index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    # Let's train :)
    train_costs_per_epoch = []
    train_costs_per_example = []
    start_time = timeit.default_timer()
    for epoch in xrange(epochs):
        plot_layers(mlp, str(epoch))
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_func(batch_index))

        train_costs_per_epoch.append(np.mean(c))
        train_costs_per_example.extend(c)
        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    return train_costs_per_epoch, train_costs_per_example


def plots(index, mlp, output_folder='plots',
          x=None, train_costs_per_epoch=None):
    # Create directory for files
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    image = Image.fromarray(
        tile_raster_images(X=mlp.weights[0].get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters.png')

    # Training curve
    fig = plt.figure()
    plt.plot(train_costs_per_epoch)
    fig.savefig('temp.png')

    # Nodes representation
    plot_layers(mlp)


def plot_layers(mlp, string='NA'):
    layer = T.matrix('layer')
    outputs = np.zeros(((mlp.num_layers - 1) / 2, 784))
    for i in np.arange((mlp.num_layers - 1) / 2, mlp.num_layers - 1):
        get_output_func = theano.function(
            [layer], mlp.get_output_from_layer(layer, i))
        # To have hot-vectors we use identity matrix
        hot_vectors = np.identity(mlp.layers[i])
        # for hot_vector in hot_vectors:
        for j in range(30):
            outputs[i - (mlp.num_layers - 1) / 2, :] = get_output_func(
                [hot_vectors[j].astype('float32')])

    image = Image.fromarray(
        tile_raster_images(X=outputs,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('/u/pezeshki/DAE_Experiments/plots/layers_epoch_' +
               string + '.png')


if __name__ == '__main__':
    index, x, mlp = build_model(
        layers=[784, 1000, 500, 250, 30, 250, 500, 1000, 784])
    train_costs_per_epoch, train_costs_per_example = train(
        index, x, mlp, corruption_level=0.0, epochs=5)
    plots(index=index, mlp=mlp, x=x, train_costs_per_epoch=train_costs_per_epoch)
    import ipdb; ipdb.set_trace()
