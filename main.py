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
        for i in np.arange(layer_number + 1, self.num_layers + 1):
            layer_ = self.get_layer_from_previous_layer(layer_, i)
        return layer_

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        z = self.get_output_from_layer(tilde_x, 0)
        loss = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(loss)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.params, gparams)]
        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=3,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='plots'):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(np_rng=rng, theano_rng=theano_rng, input=x,
            layers=[784, 20, 10, 20, 784])

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate)

    train_da = theano.function(
        [index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    h = da.get_hidden_values(x)
    z = da.get_reconstructed_input(h)
    f = theano.function(
        [index], z,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]})
    image = Image.fromarray(
        tile_raster_images(
            X=np.vstack([f(0)[:10], train_set_x.get_value()[:10]]),
            img_shape=(28, 28), tile_shape=(2, 10),
            tile_spacing=(1, 1)))
    image.save('reconstruction_corruption_0.png')

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        np_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hiddens=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))

    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')

    os.chdir('../')


if __name__ == '__main__':
    test_dA()
