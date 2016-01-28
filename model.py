import os
import random
import pickle
import datetime
from collections import defaultdict

import numpy as np

import data

def norm_weight(args, scale=0.01):
    W = scale * np.random.randn(*args)
    return W.astype('float32')

def empty_arr(shape=None):
    return np.empty(shape)

def zeros(mat, epsilon=None):
    epsilon = epsilon or 0
    return np.zeros(mat.shape) + epsilon

class Model(object):
    """
    Model is a superclass abstraction for both RNTN and RNN.

    @dim_size: dimension of the word vectors
    @num_classes: how many class output predictions we have (5).
    @step_size: adagrad step size
    @lambda_: how much to scale the L2 regularization and updates
    @batch_size: how many trees to run in each batch update
    @num_epochs: how many epochs to run through the data (25).
    @training/test/dev files: the location for these files
    @save_freq: after how many batches do we want to save the model.
    @disp_freq: after how many batches do we want to show  model results.
    @model_fin: a file location for a saved model params.
    @model_fout: a file location to save model params.
    """
    def __init__(self, dim_size, num_classes,
                 step_size=None, lambda_=None,
                 batch_size=None, num_epochs=None,
                 training_file=None, test_file=None, dev_file=None,
                 save_freq=None, disp_freq=None, model_fin=None,
                 model_fout=None):

        # Get our training / dev / test sets
        self.training_trees, tr_words = self._get_trees_from_file(training_file)
        self.dev_trees, de_words = self._get_trees_from_file(dev_file)
        self.test_trees, te_words = self._get_trees_from_file(test_file)

        words = {} # this is used to get the total vocab size
        words.update(tr_words)
        words.update(te_words)
        words.update(de_words)
        self.num_words = len(words)

        self.params, self.adagrad_gsqd = self._init_params(
            self.num_words, dim_size, num_classes, model_fin)

        self.model_fin = model_fin
        self.model_fout = model_fout or model_fin
        self.save_freq = save_freq or 50
        self.disp_freq = disp_freq or self.save_freq

        self.dim_size = dim_size
        self.num_classes = num_classes
        self.step_size = step_size or 1e-2
        self.lambda_ = lambda_ or 1e-6
        self.batch_size = batch_size or 25
        self.num_epochs = num_epochs or 25

    def grid_search(self, batch_sizes=None, dim_sizes=None):
        """Grid search over batch_size and dim_size.

        @batch_sizes: grid search over these batch sizes. ([20, 30])
        @dim_sizes: grid search over these dim_sizes ([25, 35])
        """
        batch_sizes = batch_sizes or [20, 30]
        dim_sizes = dim_sizes or [25, 35]

        best_model = None
        best_dev_accuracy = 0
        print 'Running grid search - batches:', batch_sizes, ', dim_sizes:', \
              dim_sizes

        for batch_size in batch_sizes:
            for dim_size in dim_sizes:
                print 'Starting Model batch_size:', batch_size, ', dim_size:', \
                      dim_size

                self.batch_size = batch_size
                self.dim_size = dim_size
                self.params, self.adagrad_gsqd = self._init_params(
                    self.num_words, dim_size, self.num_classes, self.model_fin)

                self.run(do_save=False)
                percent_accuracy = self._test_model(type_='dev', sample=False)
                if not best_model or percent_accuracy > best_dev_accuracy:
                    print 'New best - batch_size: %d, dim_size: %d.' % (
                        batch_size, dim_size)
                    print 'The accuracy is %.9f.' % percent_accuracy
                    best_dev_accuracy = percent_accuracy
                    best_model = {k:v for k,v in self.params.iteritems()}
                else:
                    print 'This model (%.9f) is worse than current best (%.9f)' % (
                        percent_accuracy, best_dev_accuracy)

        print 'Finished the grid search. Calculating the test accuracy:'
        self.params.update(best_model)
        test_accuracy = self._test_model(type_='test', sample=False)
        print 'Best model had dev accuracy %.9f and test accuracy %.9f' % (
            best_dev_accuracy, test_accuracy)

        return best_model

    def run(self, do_save=True):
        """Run the model. This is one entry point. Grid search is another."""

        num_batches = len(self.training_trees)/self.batch_size + 1
        last_save = 0
        last_print = 0

        for epoch in range(self.num_epochs):
            if epoch > 1:
                prev = start

            start = datetime.datetime.utcnow()
            print '\nRunning epoch #%d...' % (epoch+1)
            print 'Start Time:', str(start)
            if epoch > 1:
                print 'Time for previous epoch (%d): %s.' % (
                    epoch, str(1.0*(start - prev).seconds/60))

            for curr_batch in range(num_batches):
                print '  On batch #%d...' % (curr_batch+1)
                trees = self.training_trees[curr_batch*self.batch_size:
                                            (curr_batch+1)*self.batch_size - 1]
                cost, gradients = self._process(trees)

                # adagrad optimizer
                gradient_updates = self._adagrad(gradients)

                # update params
                self._update_parameters(gradient_updates)

                last_save += 1
                if do_save and last_save % self.save_freq == 0:
                    print '\n\nSaving model to %s.' % self.model_fout
                    self._save_model_params(self.model_fout)

                last_print += 1
                if last_print % self.disp_freq == 0:
                    print '\n\nTesting model...'
                    self._test_model(type_='dev')

    def _process(self, trees, forward_only=False):
        """Process these trees by doing a forward and backward pass.

        @trees: a batch_size list of trees given by the root.
        @forward_only: if True, then run the forward pass and don't backprop
        """
        self._reset_gradient_params()

        cost = 0.0
        success = 0
        total = 0

        for tree in trees:
            tree_cost, tree_success, tree_total = self._forward(tree)
            cost += tree_cost
            success += tree_success
            total += tree_total

        if forward_only:
            return 1.0 * cost / len(trees), success, total

        for tree in trees:
            self._backward(tree)

        # Build the updates
        gradient_ret = {}
        gradient_ret['dL'] = self.params['dL']

        ## scale the params in dL
        for k,v in gradient_ret['dL'].iteritems():
            v = 1.0 * v / self.batch_size

        ## add updates for Ws, W, and V
        for k in self._weight_keys:
            dk = 'd' + k
            gradient_ret[dk] = self.params[dk] + 2*self.lambda_*self.params[k]
            gradient_ret[dk] = 1.0 * gradient_ret[dk] / self.batch_size

        ## add updates for bs and b
        for k in self._bias_keys:
            dk = 'd' + k
            gradient_ret[dk] = 1.0 * self.params[dk] / self.batch_size

        # add L2 regularization to the cost
        for k in self._weight_keys:
            cost += self.lambda_ * np.sum(np.square(self.params[k]))
        cost = 1.0 * cost / self.batch_size

        return cost, gradient_ret

    ############
    #### Forward and Backward Calculations ####
    ############

    def _forward(self, node):
        """Do the forward pass through this node.

        @node: a root of a tree as from data.py
        """
        if node.is_leaf:
            node.h = self.params['L'][:, node.word] # dim_size x 1
            ret = [0.0, 0, 0] # cost, success, total
        else:
            left_result = self._forward(node.left)
            right_result = self._forward(node.right)
            # cost, success, total
            ret = [l+r for l,r in zip(left_result, right_result)]
            node.h = self._compute_forward_activation(node.left.h, node.right.h)

        # softmax
        node.probs = np.dot(self.params['Ws'], node.h) + self.params['bs']
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs / np.sum(node.probs)

        return [ret[0] - np.log(node.probs[node.label]), # cost
                ret[1] + int(np.argmax(node.probs) == node.label), # success
                ret[2] + 1] # total

    def _backward(self, node, error=None):
        """Do the backward pass through this node.

        @node: a root of a tree as from data.py
        @error: the deltas from the parent node. Is None if this is the root.
        """
        deltas = node.probs # num_classes x 1
        deltas[node.label] -= 1.0
        self.params['dWs'] += np.outer(deltas, node.h) # num_classes x dim_size
        self.params['dbs'] += deltas
        deltas = np.dot(self.params['Ws'].T, deltas) # dim_size x 1

        if error is not None:
            deltas += error

        deltas *= self._activation_prime(node.h) # dim_size x 1

        if node.is_leaf:
            self.params['dL'][node.word] += deltas
        else:
            down_deltas = self._compute_params_and_down_deltas(
                deltas, node.left.h, node.right.h)
            self._backward(node.left, down_deltas[:self.dim_size])
            self._backward(node.right, down_deltas[self.dim_size:])

    @staticmethod
    def _activation_func(args):
        """Tanh Activation"""
        return np.tanh(np.sum(args, axis=0))

    @staticmethod
    def _activation_prime(h):
        """Tanh Derivative"""
        return 1 - np.square(h)

    ############
    #### Optimization
    ############

    def _adagrad(self, gradients):
        """Adagrad Optimizer

        @gradients: the already calculated gradients from this batch.
        """
        e = 1e-6 # for numerical stability
        ret = {}
        for k,v in self.adagrad_gsqd.iteritems():
            if k == 'dL': # do separately
                continue
            gradient = gradients[k]
            v += np.square(gradient)
            ret[k] = 1.0 * gradient / (e + np.sqrt(v))

        key = 'dL'
        gradient = gradients[key]
        gsqd = self.adagrad_gsqd[key]
        for word, update in gradient.iteritems():
            gsqd[:, word] += np.square(gradient[word])
            gradient[word] = 1.0 * gradient[word] / (e + np.sqrt(gsqd[:, word]))
        ret[key] = gradient

        return ret

    def _update_parameters(self, gradient_updates):
        """
        Update the parameters according to the new gradients:
        e.g. V <- V + step_size * gradient_updates[dV]

        @gradient_updates: dict of key (e.g. 'dV', 'dW') to gradient update
        """
        for key, updates in gradient_updates.iteritems():
            if key == 'dL':
                continue

            self.params[key[1:]] -= self.step_size * updates

        for word, updates in gradient_updates['dL'].iteritems():
            self.params['L'] -= self.step_size * updates[...,None]

    ############
    #### Parameter Init, Reset, and Saving
    ############

    def _init_params(self, vocab_size, dim_size, num_classes):
        """Initialize the parameters given the vars.

        @vocab_size: how many words are in our vocab
        @dim_size: dimension of the word vectors
        @num_classes: how many class output predictions we have (5).
        """
        params = {}
        adagrad = {}

        # softmax W_s and b_s
        params['Ws'] = norm_weight([num_classes, dim_size])
        params['bs'] = np.zeros(num_classes).astype('float32')
        for k in ['Ws', 'bs']:
            params['d' + k] = empty_arr(params[k].shape)
            adagrad['d' + k] = zeros(params[k])

        # word embedding L
        params['L'] = norm_weight([dim_size, vocab_size], scale=0.0001)
        params['dL'] = defaultdict(lambda: np.zeros(dim_size))
        adagrad['dL'] = zeros(params['L'])

        return params, adagrad

    def _load_model_params(self, fin, curr_params=None):
        """Load model params from file.

        @fin: model file
        @curr_params: already instantiated params. replace duplicates with fin
        """
        curr_params = curr_params or {}
        if fin:
            with open(fin, 'rb') as f:
                curr_params.update(pickle.load(f))
        return curr_params

    def _save_model_params(self, fout):
        """Save the model (the weight params).

        @fout: location to save the params.
        """
        params = {k:v for k,v in self.params.iteritems() if k[0] != 'd'}
        with open(fout, 'wb') as f:
            pickle.dump(params, f)

    def _reset_gradient_params(self):
        """Reset the gradient parameters. We do this after each batch pass."""
        for key in self._gradient_keys:
            if key == 'dL':
                self.params[key] = defaultdict(lambda: np.zeros(self.dim_size))
            else:
                self.params[key][:] = 0

    ############
    #### Load the data
    ############

    def _get_trees_from_file(self, fin=None, shuffle=True):
        """Get the trees from the input file.

        @fin: input file for the PTB data.
        @shuffle: If true, randomly shuffles the data.
        """
        trees, word_dict = data.read_ptb_dataset(fin)
        if shuffle:
            random.shuffle(trees)
        return trees, word_dict

    ############
    #### Test the model
    ############

    def _test_model(self, type_=None, sample=True):
        """Test the model on data with a forward pass.

        @type_: 'dev' or 'test'.
        @sample: if True, restrict # of trees to a 4*batch_size random sample
        """
        type_ = type_ or 'test'
        assert(type_ in ['test', 'dev'])

        if type_ == 'test':
            trees = self.test_trees
        elif type_ == 'dev':
            trees = self.dev_trees

        if sample:
            trees = random.sample(trees, self.batch_size * 4)

        cost, success, total = self._process(trees, forward_only=True)
        if sample:
            print 'This sample batch had %d / %d correct (%f) on the %sset.' % (
                success, total, 1.0*success/total, type_)
        else:
            print 'This batch had %d / %d correct (%f) on the %sset.' % (
                success, total, 1.0*success/total, type_)

        print 'The cost was %.9f.\n' % cost
        return 1.0*success/total

class RNTN(Model):
    """Recursive Neural Tensor Network: Numpy Implementation

    Can call this with either .run() or .grid_search()

    as described in:
    - Recursive Deep Models for Semantic Compositionality Over a Sentiment
      Treebank (nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
    """
    def __init__(self, dim_size, num_classes,
                 step_size=None, lambda_=None,
                 batch_size=None, num_epochs=None,
                 training_file=None, test_file=None, dev_file=None,
                 save_freq=None, disp_freq=None, model_fin=None,
                 model_fout=None):
        super(RNTN, self).__init__(dim_size, num_classes, step_size, lambda_,
                                   batch_size, num_epochs,
                                   training_file, test_file, dev_file,
                                   save_freq, disp_freq, model_fin, model_fout)
        self._weight_keys = ['Ws', 'W', 'V']
        self._bias_keys = ['bs', 'b']
        self._gradient_keys = ['d'+k for k in self._weight_keys + self._bias_keys]
        self._gradient_keys.append('dL')

    def _compute_params_and_down_deltas(self, deltas, lefth, righth):
        """
        Compute the parameters and down deltas for the RNTN. Returns the
        down deltas and sets the parameters in self.

        This HAS side effects.

        @deltas: already calculated error deltas
        @lefth: the node.left.h for this node
        @righth: the node.right.h for this node
        """
        # 2*dim_size,
        concat = np.hstack([lefth, righth])

        # dim_size x 2*dim_size x 2*dim_size
        self.params['dV'] += (deltas.T * np.outer(concat, concat)[..., None]).T

        # dim_size x 2*dim_size
        self.params['dW'] += np.outer(deltas, concat)

        # dim_size,
        self.params['db'] += deltas

        # 2*dim_size,
        down_deltas = np.dot(self.params['W'].T, deltas)
        down_deltas += np.tensordot(
            self.params['V'].transpose(0,2,1) + self.params['V'],
            np.outer(deltas, concat), axes=((0, 1), (0, 1))
            )
        return down_deltas

    def _compute_forward_activation(self, lefth, righth):
        """
        Computes and returns the forward activation for the RNTN.

        This does NOT have side effects.

        @lefth: the node.left.h for this node
        @righth: the node.right.h for this node
        """
        # concat's shape is dim_size*2 x 1
        concat = np.hstack([lefth, righth])

        # compute the contribution from W and V --> dim_size,
        W_result = np.dot(self.params['W'], concat)
        V_result = np.dot(concat.T, self.params['V']).dot(concat)

        # activation function --> dim_size,
        h = self._activation_func([W_result, V_result, self.params['b']])
        h = np.squeeze(h)
        return h

    def _init_params(self, vocab_size, dim_size, num_classes, model_fin=None):
        """Init the params for the RNTN. This is separated out from the Model
        parent so that we can include the W,V,b params.

        @vocab_size: how many words are in our vocab
        @dim_size: dimension of the word vectors
        @num_classes: how many class output predictions we have (5).
        @model_fin: model file in with already saved params.
        """
        # Get the softmax and word embedding params
        params, adagrad = super(RNTN, self)._init_params(
            vocab_size, dim_size, num_classes)

        # network V, W, and b
        params['V'] = norm_weight([dim_size, dim_size*2, dim_size*2])
        params['W'] = norm_weight([dim_size, dim_size*2])
        params['b'] = np.zeros(dim_size).astype('float32')
        for k in ['V', 'W', 'b']:
            params['d' + k] = empty_arr(params[k].shape)
            adagrad['d' + k] = zeros(params[k])

        params = self._load_model_params(model_fin, params)
        return params, adagrad

    def check_gradient(self, epsilon=None):
        epsilon = epsilon or 1e-4
        trees = self.training_trees[:30]
        cost, gradients = self._process(trees)

        for key, gradient in gradients.iteritems():
            if key == 'dL':
                continue

            params = self.params[key[1:]][..., None, None]
            gradient = gradient[..., None, None]
            for i in xrange(params.shape[0]):
                for j in xrange(params.shape[1]):
                    for k in xrange(params.shape[2]):
                        params[i,j,k] += epsilon
                        pos_cost, _ = self._process(trees)
                        params[i,j,k] -= epsilon

                        numerical_gradient = 1.0 * (pos_cost - cost) / epsilon
                        difference = np.abs(gradient[i,j] - numerical_gradient)
                        relative_diff = 1.0*difference/numerical_gradient
                        print 'Our gradient: %.9f, numerical gradient: %.9f, '\
                              'difference: %.9f, relative difference %.9f' % (
                            gradient[i,j], numerical_gradient,
                            difference, relative_diff)


class RNN(Model):
    """RNN Model, simpler than the RNTN model because exclude V"""

    def __init__(self, dim_size, num_classes,
                 step_size=None, lambda_=None,
                 batch_size=None, num_epochs=None,
                 training_file=None, test_file=None, dev_file=None,
                 save_freq=None, disp_freq=None, model_fin=None,
                 model_fout=None):
        super(RNN, self).__init__(dim_size, num_classes, step_size, lambda_,
                                  batch_size, num_epochs,
                                  training_file, test_file, dev_file,
                                  save_freq, disp_freq, model_fin, model_fout)
        self._weight_keys = ['Ws', 'W']
        self._bias_keys = ['bs', 'b']
        self._gradient_keys = ['d'+k for k in
                               self._weight_keys + self._bias_keys]
        self._gradient_keys.append('dL')

    def _compute_params_and_down_deltas(self, deltas, lefth, righth):
        """Compute the parameters and down deltas for the RNN. Returns the
        down deltas and sets the parameters in self. This HAS side effects.

        @deltas: already calculated error deltas
        @lefth: the node.left.h for this node
        @righth: the node.right.h for this node
        """
        # 2*dim_size,
        concat = np.hstack([lefth, righth])

        # dim_size x 2*dim_size
        self.params['dW'] += np.outer(deltas, concat)

        # dim_size,
        self.params['db'] += deltas

        # 2*dim_size,
        down_deltas = np.dot(self.params['W'].T, deltas)
        return down_deltas

    def _compute_forward_activation(self, lefth, righth):
        """Computes and returns the forward activation for the RNN. This does
        NOT have side effects.

        @lefth: the node.left.h for this node
        @righth: the node.right.h for this node
        """
        # concat's shape is dim_size*2 x 1
        concat = np.hstack([lefth, righth])

        # compute the contribution from W and V --> dim_size,
        W_result = np.dot(self.params['W'], concat)

        # activation function --> dim_size,
        h = self._activation_func([W_result, self.params['b']])
        h = np.squeeze(h)
        return h

    def _init_params(self, vocab_size, dim_size, num_classes, model_fin=None):
        """Init the params for the RNTN. This is separated out from the Model
        parent so that we can include the W,b params.

        @vocab_size: how many words are in our vocab
        @dim_size: dimension of the word vectors
        @num_classes: how many class output predictions we have (5).
        @model_fin: model file in with already saved params.
        """
        # Get the softmax and word embedding params
        params, adagrad = super(RNN, self)._init_params(
            vocab_size, dim_size, num_classes)

        # network W, and b
        params['W'] = norm_weight([dim_size, dim_size*2])
        params['b'] = np.zeros(dim_size).astype('float32')
        for k in ['W', 'b']:
            params['d' + k] = empty_arr(params[k].shape)
            adagrad['d' + k] = zeros(params[k])

        params = self._load_model_params(model_fin, params)
        return params, adagrad

    def check_gradient(self, epsilon=None):
        epsilon = epsilon or 1e-4
        trees = self.training_trees[:30]
        cost, gradients = self._process(trees)

        for key, gradient in gradients.iteritems():
            if key == 'dL':
                continue

            params = self.params[key[1:]][..., None]
            gradient = gradient[..., None]
            for i in xrange(params.shape[0]):
                for j in xrange(params.shape[1]):
                    params[i,j] += epsilon
                    pos_cost, _ = self._process(trees)
                    params[i,j] -= epsilon

                    numerical_gradient = 1.0 * (pos_cost - cost) / epsilon
                    difference = np.abs(gradient[i,j] - numerical_gradient)
                    relative_diff = 1.0*difference/numerical_gradient
                    print 'Our gradient: %.9f, numerical gradient: %.9f, '\
                          'difference: %.9f, relative difference %.9f' % (
                        gradient[i,j], numerical_gradient,
                        difference, relative_diff)
