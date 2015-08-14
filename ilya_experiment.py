#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Yann Dauphin on 2013-09-29.
Copyright (c) 2013 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import cPickle
import gzip
import time

import numpy
import scipy.optimize
import scipy.sparse
from scipy.sparse.linalg import LinearOperator, cg
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams


import theano
from theano import tensor as T
import jobman
from jobman.tools import DD


theano.config.floatX = 'float32'


def main(n_inputs=784,
         n_hiddens0=1000,
         n_hiddens1=500,
         n_hiddens2=250,
         n_hiddens3=30,
         momentum=0.995,
         learning_rate=0.0001,
         n_updates=750000,
         batch_size=200,
         restart=0,
         state=None,
         channel=None,
         **kwargs):
    numpy.random.seed(0xeffe)

    print locals()
    print "Tags: fixed-init"

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/data/lisatmp/dauphiya/ddbm/mnist.pkl.gz", 'rb'))

    inds = range(train_x.shape[0])
    numpy.random.shuffle(inds)
    train_x = numpy.ascontiguousarray(train_x[inds])

    s_train_x = theano.shared(train_x)

    def init_param(nx, ny, name=None):
        W = numpy.zeros((nx,ny), dtype='float32')
        scale=1.
        if name=='56':
            scale=1.
        for j in xrange(ny):
            perm = numpy.random.permutation(nx)
            for k in xrange(numpy.min([15, nx])):
                W[perm[k], j] = numpy.random.normal(loc=0, scale=1.)*scale
        return theano.shared(W)

    W0 = init_param(n_inputs, n_hiddens0)
    W1 = init_param(n_hiddens0, n_hiddens1)
    W2 = init_param(n_hiddens1, n_hiddens2)
    W3 = init_param(n_hiddens2, n_hiddens3)
    W0_ = init_param(n_hiddens0, n_inputs)
    W1_ = init_param(n_hiddens1, n_hiddens0)
    W2_ = init_param(n_hiddens2, n_hiddens1)
    W3_ = init_param(n_hiddens3, n_hiddens2)
    b0 = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b1 = theano.shared(numpy.zeros(n_hiddens1, 'float32'))
    b2 = theano.shared(numpy.zeros(n_hiddens2, 'float32'))
    b3 = theano.shared(numpy.zeros(n_hiddens3, 'float32'))
    b0_ = theano.shared(numpy.zeros(n_inputs, 'float32'))
    b1_ = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b2_ = theano.shared(numpy.zeros(n_hiddens1, 'float32'))
    b3_ = theano.shared(numpy.zeros(n_hiddens2, 'float32'))
    params = [W0, b0, W1, b1, W2, b2, W3, b3, W0_, b0_, W1_, b1_, W2_, b2_, W3_, b3_]

    input = T.matrix('x')
    index = T.lscalar('i')
    timestep = T.lscalar('t')
    vec = T.vector('v')

    hidden = T.nnet.sigmoid(T.dot(input, W0) + b0)
    hidden = T.nnet.sigmoid(T.dot(hidden, W1) + b1)
    hidden = T.nnet.sigmoid(T.dot(hidden, W2) + b2)
    hidden = T.dot(hidden, W3) + b3

    hidden = T.nnet.sigmoid(T.dot(hidden, W3_) + b3_)
    hidden = T.nnet.sigmoid(T.dot(hidden, W2_) + b2_)
    hidden = T.nnet.sigmoid(T.dot(hidden, W1_) + b1_)
    output_pre = T.dot(hidden, W0_) + b0_
    output = T.nnet.sigmoid(output_pre)

    loss = -(input * T.log(output) + (1. - input) * T.log(1. - output)).sum(1).mean()
    gparams = T.grad(loss, params)

    givens = {
        input : s_train_x[index * batch_size:(index + 1) * batch_size],
    }

    n_batches = len(train_x) / batch_size

    #mu = T.switch(T.lt(timestep, n_updates - 20 * n_batches), momentum, momentum/2)
    mu = T.switch(T.lt(timestep, n_updates - 4 * n_batches), T.cast(T.clip(1 - 2**(-1 - T.log2(T.floor(timestep / float(n_batches)) + 1)), 0, momentum), "float32"), 0.9)

    updates = []
    memories = []
    for param, gparam in zip(params, gparams):
        memory = theano.shared(param.get_value() * 0.)
        update = mu * memory - learning_rate * gparam
        updates.append((memory, update))
        updates.append((param, param + update))
        memories.append(memory)
    train = theano.function([timestep, index], loss, givens=givens, updates=updates)

    updates = []
    for param, memory in zip(params, memories):
        updates.append((param, param + mu * memory))
    update_param = theano.function([timestep, index], loss, givens=givens, updates=updates)

    loss = theano.function([input], loss)
    output = theano.function([input], output)

    begin = time.time()
    best_train_error = float("inf")

    if restart:
        saved_params = numpy.load("/data/lisatmp/dauphiya/saddle_mnist_ae/dauphiya_db/saddle_mnist_ae/%d/params.npy" % restart)
        for param, value in zip(params, saved_params):
            param.set_value(value)

    print "Training..."
    for update in range(n_updates):
        update_param(update, update % n_batches)
        train(update, update % n_batches)

        if update % (10 * n_batches) == 0 or update == n_updates - 1:
            if state == None:
                numpy.save("params_%d.npy" % (update / n_batches), [param.get_value() for param in params])
            
            train_error = ((output(train_x) - train_x)**2).sum(1).mean()

            if train_error < best_train_error:
                best_train_error = train_error

            if state != None:
                state.train_error = best_train_error

                if channel != None:
                    channel.save()

            print  "[%d, %f, %f, %f]," % (update / n_batches, time.time() - begin, loss(train_x), train_error)

    numpy.save("params.npy", [param.get_value() for param in params])


def jobman_entrypoint(state, channel):
    main(state=state, channel=channel, **state)

    return channel.COMPLETE

def jobman_insert_random(n_jobs):
    JOBDB = 'postgres://dauphiya:wt17se79@opter.iro.umontreal.ca/dauphiya_db/saddle_mnist_ae'
    EXPERIMENT_PATH = "ilya_experiment.jobman_entrypoint"

    jobs = []
    for _ in range(n_jobs):
        job = DD()

        job.learning_rate = 10.**numpy.random.uniform(-2, 0)
        job.momentum = 10.**numpy.random.uniform(-2, 0)
        job.batch_size = 200
        job.tag = "ilya_fixed"

        jobs.append(job)
        print job

    answer = raw_input("Submit %d jobs?[y/N] " % len(jobs))
    if answer == "y":
        numpy.random.shuffle(jobs)

        db = jobman.sql.db(JOBDB)
        for job in jobs:
            job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
            jobman.sql.insert_dict(job, db)

        print "inserted %d jobs" % len(jobs)
        print "To run: jobdispatch --condor --gpu --env=THEANO_FLAGS='floatX=float32, device=gpu' --repeat_jobs=%d jobman sql -n 1 'postgres://dauphiya:wt17se79@opter.iro.umontreal.ca/dauphiya_db/saddle_mnist_ae' ." % len(jobs)


def view(table="saddle_mnist_ae",
         tag="rbm1",
         user="dauphiya",
         password="wt17se79",
         database="dauphiya_db",
         host="opter.iro.umontreal.ca"):
    """
    View all the jobs in the database.
    """
    import commands
    import sqlalchemy
    import psycopg2

    # Update view
    url = "postgres://%s:%s@%s/%s/" % (user, password, host, database)
    commands.getoutput("jobman sqlview %s%s %s_view" % (url, table, table))

    # Display output
    def connect():
        return psycopg2.connect(user=user, password=password,
                                database=database, host=host)

    engine = sqlalchemy.create_engine('postgres://', creator=connect)
    conn = engine.connect()
    experiments = sqlalchemy.Table('%s_view' % table,
                                   sqlalchemy.MetaData(engine), autoload=True)

    columns = [experiments.columns.id,
               experiments.columns.jobman_status,
               experiments.columns.tag,
               experiments.columns.nhiddens0,
               experiments.columns.learningrate,
               experiments.columns.momentum,
               experiments.columns.batchsize,
               experiments.columns.method,
               experiments.columns.trainerror,]

    results = sqlalchemy.select(columns,
                                order_by=[experiments.columns.tag,
                                    sqlalchemy.desc(experiments.columns.trainerror)]).execute()
    results = [map(lambda x: x.name, columns)] + list(results)

    def get_max_width(table, index):
        """Get the maximum width of the given column index"""
        return max([len(format_num(row[index])) for row in table])

    def format_num(num):
        """Format a number according to given places.
        Adds commas, etc. Will truncate floats into ints!"""
        try:
            if "." in num:
                return "%.7f" % float(num)
            else:
                return int(num)
        except (ValueError, TypeError):
            return str(num)

    col_paddings = []

    for i in range(len(results[0])):
        col_paddings.append(get_max_width(results, i))

    for row_num, row in enumerate(results):
        for i in range(len(row)):
            col = format_num(row[i]).ljust(col_paddings[i] + 2) + "|"
            print col,
        print

        if row_num == 0:
            for i in range(len(row)):
                print "".ljust(col_paddings[i] + 1, "-") + " +",
            print

if __name__ == "__main__":
    if "insert" in sys.argv:
        jobman_insert_random(int(sys.argv[2]))
    elif "view" in sys.argv:
        view()
    else:
        main()
