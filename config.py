import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_string('dataset', 'mnist', 'name of dataset [mnist, fashion_mnist]')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('batch_size', 50, 'batch size')
flags.DEFINE_integer('epochs', 100, 'number of epochs')
flags.DEFINE_boolean('use_checkpoint', True, 'restore model from checkpoint')
flags.DEFINE_boolean('extended_dataset', True, 'use extended models (with rotations and repositioning)')


cfg = tf.app.flags.FLAGS