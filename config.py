import tensorflow as tf

flags = tf.app.flags


#####################
#command prompt flags
#####################
flags.DEFINE_string('dataset', 'mnist', 'name of dataset [mnist]')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('batch_size', 50, 'batch size')
flags.DEFINE_boolean('use_checkpoint', True, 'restore model from checkpoint' )

cfg = tf.app.flags.FLAGS