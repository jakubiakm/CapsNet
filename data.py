import numpy
import gzip
import tensorflow as tf

class Data(object):
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self._train = DataSet(train_images, train_labels)
        self._test = DataSet(test_images, test_labels)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
    
    @property
    def validation(self):
        return self._test

class DataSet(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]


    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
                (images_rest_part, images_new_part), axis=0), numpy.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def load_data(dataset):
    if dataset == 'mnist':
        return load_mnist()
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)

def load_mnist():
    img_size = 784
    train_images, train_labels = tf.keras.datasets.mnist.load_data()[0]
    test_images, test_labels = tf.keras.datasets.mnist.load_data()[1]
    train_images = train_images.reshape(-1, img_size)
    train_images = numpy.multiply(train_images, 1.0 / 255.0)
    test_images = test_images.reshape(-1, img_size)
    test_images = numpy.multiply(test_images, 1.0 / 255.0)
    return Data(train_images, train_labels, test_images, test_labels)