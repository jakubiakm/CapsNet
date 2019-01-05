import numpy
import gzip
import tensorflow as tf
from keras.datasets import fashion_mnist, mnist
from scipy.ndimage import rotate, shift
from math import sqrt

class Data(object):
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self._train = DataSet(train_images, train_labels)
        self._test = DataSet(test_images, test_labels)

        assert self._train.images[0].size == self._test.images[0].size, \
        ('train_images.shape: %s test_images.shape: %s' % (self._train.images.shape, self._test.images.shape))
        # silently we think here that all images have the same x and y dimension length
        self._image_axis_size = int(sqrt(self._train.images[0].size))
    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
    
    @property
    def validation(self):
        return self._test

    @property
    def image_axis_size(self):
        return self._image_axis_size

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

def load_data(cfg):
    dataset_type = cfg.dataset
    extended_dataset = cfg.extended_dataset
    if dataset_type in ('mnist', 'fashion_mnist'):
        if extended_dataset:
            extension_type = cfg.extension_type
            return load_extended_dataset(dataset_type, extension_type)
        else:
            return load_dataset(dataset_type)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset_type)


def get_dataset_from_type(dataset_type):
    if dataset_type == "mnist":
        return mnist.load_data()
    elif dataset_type == "fashion_mnist":
        return fashion_mnist.load_data()
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset_type)


def reshape_normalize_dataset(dataset, array_size):
    dataset = dataset.reshape(-1, array_size)
    dataset = numpy.multiply(dataset, 1.0 / 255.0)

    return dataset


def load_dataset(dataset_type):
    dataset = get_dataset_from_type(dataset_type)

    # train data -> data -> first element
    img_size = dataset[0][0][0].size

    train_images, train_labels = dataset[0]
    test_images, test_labels = dataset[1]

    train_images = reshape_normalize_dataset(train_images, img_size)
    test_images = reshape_normalize_dataset(test_images, img_size)

    return Data(train_images, train_labels, test_images, test_labels)


def get_parameters_for_extension_type(extension_type):
    multiplier = {'rotation': 3, 'shift': 5, 'both': 8, 'none': 1}

    return multiplier[extension_type]


def crop_image(image, axis_size):
    return image[0:axis_size, 0:axis_size]


def get_nth_image_from_operation(image, label, extension_type, j):
    axis_size = image.shape[0]
    rotations = [0.0, -5.0, 5.0]
    shifts = [(-5, 0), (5, 0), (0, 0), (0, -5), (0, 5)]

    out_image = None

    if (extension_type == "rotation" and j > len(rotations) - 1) or \
            (extension_type == "shift" and j > len(shifts) - 1):
        raise ValueError("Index not suitable for given operation!")

    if extension_type == "both":
        if j < 3:
            out_image = rotate(image, rotations[j])
        elif j >= 3:
            out_image = shift(image, shifts[j - 3])
        else:
            raise ValueError("Invalid index!")
    elif extension_type == "rotation":
        out_image = rotate(image, rotations[j])
    elif extension_type == "shift":
        out_image = shift(image, shifts[j])
    else:
        raise ValueError("Invalid extension type!")

    return crop_image(out_image, axis_size), label


def extend_datasets_by_operation(images, labels, extension_type="none"):
    multiplier = get_parameters_for_extension_type(extension_type)

    extended_shape = list()
    extended_shape.append(images.shape[0] * multiplier)
    extended_shape = extended_shape + list(images.shape[1:])

    new_images = numpy.zeros(extended_shape)
    new_labels = numpy.zeros(extended_shape[0], dtype='uint8')

    for i in range(0, images.shape[0]):
        for j in range(0, multiplier):
            new_images[3 * i + j], new_labels[3 * i + j] = \
                get_nth_image_from_operation(images[i], labels[i], extension_type, j)

    return new_images, new_labels


def print_info_about_dataset(dataset):
    print(dataset.shape)


def print_info_about_datasets(train_images, train_labels, test_images, test_labels):
    print_info_about_dataset(train_images)
    print_info_about_dataset(train_labels)
    print_info_about_dataset(test_images)
    print_info_about_dataset(test_labels)


def load_extended_dataset(dataset_type, extension_type):
    dataset = get_dataset_from_type(dataset_type)
    train_images, train_labels = dataset[0]
    test_images, test_labels = dataset[1]

    img_size = train_images[0].size

    if extension_type and not extension_type == "none":
        print("Applying extensions...")
        train_images, train_labels = extend_datasets_by_operation(train_images, train_labels, extension_type)
        test_images, test_labels = extend_datasets_by_operation(test_images, test_labels, extension_type)
        print("Applying extensions done...")

    print_info_about_datasets(train_images, train_labels, test_images, test_labels)

    train_images = reshape_normalize_dataset(train_images, img_size)
    test_images = reshape_normalize_dataset(test_images, img_size)

    return Data(train_images, train_labels, test_images, test_labels)