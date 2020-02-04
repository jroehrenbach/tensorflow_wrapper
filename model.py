# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:38:01 2020

@author: jroeh
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def cross_entropy_loss(y, y_pred):
    return -tf.reduce_sum(y*tf.log(y_pred)+(1-y)*tf.log(1-y_pred), axis=1)


class Layer(object):
    
    def __init__(self, neurons):
        self.neurons = neurons


class Input(Layer):
    """input layer with input placeholder"""
    
    def __init__(self, x):
        Layer.__init__(self, x.shape[1])
        self.out = x


class FeedForward(Layer):
    """layer object"""
    
    def __init__(self, previous, neurons, activation, set_biases=True):
        """
        previous : Layer
        neurons : int
            Number of neurons
        activation : str or tensorflow function
            Name of tensorflow function or tensorflow function
        set_biases : bool (optional)
            If true, biases will be set
        """
        Layer.__init__(self, neurons)
        
        # set weights and biases
        w = tf.random.uniform([previous.neurons, neurons], 0, 1)
        self.weights = tf.cast(tf.Variable(w), tf.float64)
        if set_biases:
            b = tf.random.uniform([1, neurons], 0, 1)
            self.biases = tf.cast(tf.Variable(b), tf.float64)
        
        # setup feed forward
        if type(activation) == str:
            activation = getattr(tf, activation)
        _pred = tf.matmul(previous.out, self.weights)
        pred = tf.add(_pred, self.biases) if set_biases else _pred
        self.out = activation(pred)


class Model(object):
    """tensorflow wrapper for feed forward model"""
    
    def __init__(self, x, y, test_size=0.2):
        """
        Parameters
        ----------
        x, y : numpy array
            Data set will be used for training and testing
        learning_rate : number
        test_size : number, optional
            Ratio of data being used for training and for testing
        """
        test_idx = int(x.shape[0] * (1-test_size))
        self.x_train = x[:test_idx]
        self.y_train = y[:test_idx]
        self.x_test = x[test_idx:]
        self.y_test = y[test_idx:]
        
        self.x = tf.placeholder(tf.float64, shape=[None, x.shape[1]])
        self.y = tf.placeholder(tf.float64, shape=[None, y.shape[1]])
        self.lr = tf.placeholder(tf.float64, shape=[])
        self.layers = [Input(self.x)]
    
    
    def add_feed_forward_layer(self, neurons, activation, set_biases=True):
        """
        Adds feed forward layer to model
        
        Parameters
        ----------
        neurons : int
            Number of neurons
        activation : str or tensorflow function
            Name of tensorflow function or tensorflow function
        set_biases : bool (optional)
            If true, biases will be set
        """
        ff = FeedForward(self.layers[-1], neurons, activation, set_biases)
        self.layers.append(ff)
    
    
    def compile_backprop(self, loss_func, optimizer="GradientDescentOptimizer"):
        """
        Sets loss function and initializes optimizer
        
        Paramerters
        -----------
        loss_func : function
        optimizer : str (optional)
            Name of tensorflow optimizer (GradientDescentOptimizer by default)
        """
        self.y_pred = self.layers[-1].out
        
        # calculating loss function
        loss = loss_func(self.y, self.y_pred)
        
        # setup backpropagation
        _opt = getattr(tf.train, optimizer)(learning_rate = self.lr)
        self.optimizer = _opt.minimize(loss)

    
    
    def start_session(self, ckpt_path=""):
        """
        Initializes the global variables and opens a new tensorflow session if
        no checkpoint is passed, otherwise it opens a saved session.
        
        Parameters
        ----------
        ckpt_path : str, optional
            directory path for a previously saved session
        """
        
        if not hasattr(self, "y_pred"):
            raise IOError("model has not been compiled!")
        
        self.sess = tf.Session()
        if ckpt_path=="":
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print("new session has been created")
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_path)
            print("session has been loaded from", ckpt_path)
    
    
    def save_session(self, ckpt_path):
        """
        Saves the current tensorflow session to the path indicated
        
        Parameters
        ----------
        ckpt_path : str
            path for current session to be saved to
        """

        saver = tf.train.Saver()
        saver.save(self.sess, ckpt_path)
    
    
    def feed(self, x, y, fetches, learning_rate=0):
        """
        Runs session and returns fetches in a dataframe
        
        Paramters
        ---------
        x, y : np.ndarray
            Data set which is fed to the model
        fetches : list of tf.Tensor
            List of tensorflow objects to be returned by the model
        learning_rate : float
        
        Returns
        -------
        pd.DataFrame with fetches
        """
        
        if fetches == []:
            fetches = [self.y_pred]
        
        feed_dict = {self.x:x, self.y:y}
        if learning_rate>0:
            fetches.append(self.optimizer)
            feed_dict[self.lr] = learning_rate
        
        vals = self.sess.run(fetches, feed_dict=feed_dict)
        if learning_rate>0:
            del vals[-1]
        
        out = pd.DataFrame()
        for fetch, val in zip(fetches, vals):
            if type(val) != np.ndarray:
                continue
            out[fetch.name] = val.ravel()
        
        return out
    
    
    def train_epoch(self, learning_rate, batchsize, fetches=[]):
        """
        Uses training data set to train one epoch devided into batches and
        returns requested model parameters as data frame
        
        Parameters
        ----------
        learning_rate : float
        batchsize : int
            Has to be smaller than training data set
        fetches : list of tf.Tensor
            List of tensorflow objects to be returned by the model
        
        Returns
        -------
        pd.DataFrame with all fetches
        """
        
        model_output = pd.DataFrame()
        total_batch = int(self.x_train.shape[0] / batchsize)
        for batch in range(total_batch):
            
            batch_idx = np.random.choice(self.x_train.shape[0], batchsize)
            x, y = self.x_train[batch_idx], self.y_train[batch_idx]
            
            _out = self.feed(x, y, fetches, learning_rate)
            out = model_output.append(_out, sort=False, ignore_index=True)
        
        return out
    
    
    def train(self, learning_rate, batchsize, epochs, fetches=[]):
        """
        Trains model on training dataset
        
        learning_rate : float
        batchsize : int
        epochs : int
        fetches : list of tf.Tensor
            List of tensorflow objects to be returned by the model
        
        Returns
        -------
        pd.DataFrame with statistics derived from the fetches
        """
        
        stats = pd.DataFrame()
        for epoch in range(epochs):
            
            out = self.train_epoch(learning_rate, batchsize, fetches)
            stats = stats.append(out.mean(), sort=False, ignore_index=True)
        
        stats.index.name = "epoch"
        return stats
    
    
    def predict(self, x):
        """
        Predicts labels for x
        
        Parameters
        ----------
        x : np.ndarray
            Numpy array with features
        
        Returns
        -------
        np.ndarray with labels
        """
        return self.sess.run(self.y_pred, {self.x: x})
    
    
    def get_weights(self, layer_index):
        """
        Returns weights of layer
        
        Parameters
        ----------
        layer_index : int
            Index of layer with weights
        
        Returns
        -------
        np.ndarray of weights
        """
        if not hasattr(self.layers[layer_index], "weights"):
            raise IOError("Layer doesn't have weights!")
        return self.sess.run(self.layers[layer_index].weights)
    
    
    def get_biases(self, layer_index):
        """
        Returns biases of layer
        
        Parameters
        ----------
        layer_index : int
            Index of layer with biases
        
        Returns
        -------
        np.ndarray of biases
        """
        if not hasattr(self.layers[layer_index], "biases"):
            raise IOError("Layer doesn't have biases!")
        return self.sess.run(self.layers[layer_index].biases)


def xor_test():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    
    model = Model(X, Y, 0)
    model.add_feed_forward_layer(2, "sigmoid")
    model.add_feed_forward_layer(1, "sigmoid")
    model.compile_backprop(cross_entropy_loss)
    model.start_session()
    
    p_pos = tf.math.greater(model.y_pred, 0.5, "positive")
    p_neg = tf.logical_not(p_pos, "negative")
    y_pos = tf.math.equal(model.y, 1, "truth")
    acc = tf.math.equal(p_pos, y_pos, "accuracy")
    tp = tf.logical_and(acc, p_pos, "true_positive")
    tn = tf.logical_and(acc, p_neg, "true_negative")
    
    print(model.feed(X,Y,[model.y, model.y_pred]))
    model.train(0.2, 4, 1000, [acc, tp, tn])
    print(model.feed(X,Y,[model.y, model.y_pred]))


if __name__ == "__main__":
    xor_test()