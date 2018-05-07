import tensorflow as tf

from config import cfg

def main(_):
    tf.logging.info('Hello world...')
    print(cfg.dataset)

if __name__ == "__main__":
    tf.app.run()