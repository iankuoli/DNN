__author__ = 'iankuoli'

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

class Layer(object):

    def __init__(self, rng, input, num_in, num_out, mat_W=None, vec_b=None, activation=T.tanh):

        self.input = input
        self.vec_Delta = T.matrix('mat_Delta')
        self.vec_Sigma = T.vector('vec_Sigma')

        if mat_W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (num_in + num_out)),
                    high=numpy.sqrt(6. / (num_in + num_out)),
                    size=(num_in, num_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            mat_W = theano.shared(value=W_values, name='mat_W', borrow=True)

        if vec_b is None:
            b_values = numpy.zeros((num_out,), dtype=theano.config.floatX)
            vec_b = theano.shared(value=b_values, name='vec_b', borrow=True)

        # parameters of the layer
        self.mat_W = mat_W
        self.vec_b = vec_b

        # linear combination
        self.vec_z = T.dot(input, self.mat_W) + self.vec_b

        # output vector by an activation function (tanh or sigmoid)
        self.vec_a = (
            self.vec_z if activation is None
            else activation(self.vec_z)
        )

        # parameters of the model
        self.params = [self.mat_W, self.vec_b]