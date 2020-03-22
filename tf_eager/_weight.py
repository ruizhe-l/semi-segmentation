import numpy as np
import tensorflow as tf
    
def balance_weight_map(flat_labels):
    '''
    return the balance weight map in 1-D tensor
    :param flat_labels: masked ground truth tensor in shape [-1, n_class]
    '''
    n = tf.shape(flat_labels)[0]
    c = 1/(tf.reduce_sum(1/(tf.reduce_sum(flat_labels, axis=0))))
    weight_map = tf.reduce_sum(tf.multiply(flat_labels, tf.tile(c/(tf.reduce_sum(flat_labels, axis=0, keepdims=True)), [n, 1])), axis=-1)
    # weight_map = tf.clip_by_value(weight_map,1e-10,1.0)
    # weight_map = nan_to_zero(weight_map)
    return tf.cast(weight_map, tf.float32) 

def feedback_weight_map(flat_probs, flat_labels, beta, op):
    '''
    return the feedback weight map in 1-D tensor
    :param flat_probs: prediction tensor in shape [-1, n_class]
    :param flat_labels: ground truth tensor in shape [-1, n_class]
    '''
    probs = tf.reduce_sum(flat_probs*flat_labels, axis=-1)
    weight_map = tf.exp(-tf.pow(probs, beta)*tf.log(tf.constant(op, "float")))   
    # weight_map = tf.clip_by_value(weight_map,1e-10, 1.0)
    weight_map = nan_to_zero(weight_map)
    return weight_map 

def confidence_weight_map(flat_probs, flat_labels, beta):
    # y = 1. / (1 - log(power(x, 15)))
    flat_probs = np.array(flat_probs)
    flat_labels = np.array(flat_labels)
    probs = np.sum(flat_probs*flat_labels, axis=-1)
    weight_map = 1 / (1 - np.log(np.power(probs, beta)))
    # probs[probs < 0.6] = 0
    weight_map = probs
    
    # weight_map = tf.clip_by_value(weight_map,1e-10, 1.0)
    # weight_map = nan_to_zero(weight_map)  
    return tf.constant(weight_map, tf.float32) 

def nan_to_zero(nan_map):
    nan_map = tf.where(tf.is_nan(nan_map), tf.zeros_like(nan_map), nan_map)
    return nan_map