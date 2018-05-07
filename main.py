import tensorflow as tf
import logging


from config import cfg
from utils import load_data

def main(_):
    data = load_data(cfg.dataset, cfg.batch_size, True)

if __name__ == "__main__":
    tf.app.run()