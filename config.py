import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_string('dataset', 'mnist', 'name of dataset [mnist]')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')

cfg = tf.app.flags.FLAGS