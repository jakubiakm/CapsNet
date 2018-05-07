import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(batch_size, is_training=True):
    return input_data.read_data_sets("/tmp/data/")
    
def load_data(dataset, batch_size, is_training=True):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)
