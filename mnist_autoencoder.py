#!/usr/bin/env python

import logging
import numpy as np

from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale, Adasecant, AdaDelta, Momentum
from blocks.bricks import MLP, WEIGHT, Logistic
from blocks.bricks.cost import SquaredError, BinaryCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant, Sparse
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop


def create_model():
    """Create the deep autoencoder model with Blocks, and load MNIST."""
    mlp = MLP(activations=[Logistic(), Logistic(), Logistic(), None,
                           Logistic(), Logistic(), Logistic(), Logistic()],
              dims=[784, 1000, 500, 250, 30, 250, 500, 1000, 784],
              weights_init=Sparse(15, IsotropicGaussian()),
              biases_init=Constant(0))
    mlp.initialize()

    x = tensor.matrix('features')
    x_hat = mlp.apply(tensor.flatten(x, outdim=2))
    squared_err = SquaredError().apply(tensor.flatten(x, outdim=2), x_hat)
    cost = BinaryCrossEntropy().apply(tensor.flatten(x, outdim=2), x_hat)

    return x, cost, squared_err


def main(save_to, num_epochs, bokeh=False, step_rule=False,
         batch_size=200, lambda1=0.0, lambda2=0.0):

    random_seed = 0xeffe

    x, cost, squared_err = create_model()

    cg = ComputationGraph([cost])
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)

    if lambda1 > 0.0:
        cost += lambda1 * sum([w.__abs__().sum() for w in weights])

    if lambda2 > 0.0:
        cost += lambda2 * sum([(w ** 2).sum() for w in weights])

    cost.name = 'final_cost'

    mnist_train = MNIST("train", sources=['features'])
    mnist_test = MNIST("test", sources=['features'])

    if step_rule == 'sgd':
        step_rule = Momentum(learning_rate=0.01, momentum=.95)
        print "Using vanilla SGD"
    elif step_rule == 'adadelta':
        step_rule = AdaDelta(decay_rate=0.95)
        print "Using Adadelta"
    else:
        step_rule = Adasecant(delta_clip=25, use_adagrad=True)
        print "Using Adasecant"

    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=step_rule)
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost],
                      Flatten(
                          DataStream.default_stream(
                              mnist_test,
                              iteration_scheme=SequentialScheme(
                                  mnist_test.num_examples, 500)),
                          which_sources=('features',)),
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, squared_err,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to, save_separately=['log', '_model']),
                  Printing()]

    if bokeh:
        extensions.append(
            Plot('MNIST Autoencoder',
                 channels=[
                    ['test_final_cost'],
                    ['train_total_gradient_norm']], after_epoch=True))

    main_loop = MainLoop(
        algorithm,
        Flatten(
            DataStream.default_stream(
                mnist_train,
                iteration_scheme=ShuffledScheme(
                    mnist_train.num_examples, batch_size=batch_size,
                    rng=np.random.RandomState(random_seed))),
            which_sources=('features',)),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Training an Autoencoder on MNIST.")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist_ae.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    parser.add_argument("--bokeh", action='store_true',
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--step-rule", default='Adasecant',
                        help="Optimizer")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Batch size.")
    args = parser.parse_args()
    main(args.save_to, args.num_epochs, args.bokeh, args.step_rule,
         args.batch_size)

