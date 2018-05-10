from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils as u

from capsNet import CapsNet

from config import cfg




def main(_):
    data = u.load_data(cfg.dataset, cfg.batch_size, True)
    model = CapsNet()
    n_epochs = 100
    
    n_iterations_per_epoch = data.train.num_examples // cfg.batch_size
    n_iterations_validation = data.validation.num_examples // cfg.batch_size
    if(cfg.is_training):
        train(True, n_epochs, n_iterations_per_epoch, n_iterations_validation, data, model)  
    else:
        validate(data, model)
    #show_images(data, model)
    


def train(restore_checkpoint, n_epochs, n_iterations_per_epoch, n_iterations_validation, data, model):
    with tf.Session() as sess:
        best_loss_val = np.infty
        saver = tf.train.Saver()
        checkpoint_path = u.get_checkpoint_path()
        init = tf.global_variables_initializer()
        if cfg.use_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()

        for epoch in range(n_epochs):
            for iteration in range(1, n_iterations_per_epoch + 1):
                X_batch, y_batch = data.train.next_batch(cfg.batch_size)
                # Run the training operation and measure the loss:
                _, loss_train = sess.run(
                    [model.training_op, model.loss],
                    feed_dict={model.X: X_batch.reshape([-1, 28, 28, 1]),
                            model.y: y_batch,
                            model.mask_with_labels: True})
                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                        iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train),
                    end="")

            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            for iteration in range(1, n_iterations_validation + 1):
                X_batch, y_batch = data.validation.next_batch(cfg.batch_size)
                loss_val, acc_val = sess.run(
                        [model.loss, model.accuracy],
                        feed_dict={model.X: X_batch.reshape([-1, 28, 28, 1]),
                                model.y: y_batch})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                    end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

            # And save the model if it improved:
            if loss_val < best_loss_val:
                save_path = saver.save(sess, checkpoint_path)
                best_loss_val = loss_val

def validate(data, model):
    n_iterations_test = data.test.num_examples // cfg.batch_size
    saver = tf.train.Saver()
    checkpoint_path = u.get_checkpoint_path()
        
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = data.test.next_batch(cfg.batch_size)
            loss_test, acc_test = sess.run(
                    [model.loss, model.accuracy],
                    feed_dict={model.X: X_batch.reshape([-1, 28, 28, 1]),
                            model.y: y_batch})
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                    iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))

def show_images(data, model):
    checkpoint_path = u.get_checkpoint_path()
        

    n_samples = 5

    sample_images = data.test.images[:n_samples].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        model.saver.restore(sess, checkpoint_path)

        caps2_output_value, decoder_output_value, y_pred_value = sess.run(
                [model.caps2_output, model.decoder_output, model.y_pred],
                feed_dict={model.X: sample_images,
                        model.y: np.array([], dtype=np.int64)})
        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(1, n_samples, index + 1)
            plt.imshow(sample_images[index], cmap="binary")
            plt.title("L:" + str(data.test.labels[index]))
            plt.axis("off")

        plt.show()

        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(1, n_samples, index + 1)
            plt.title("P:" + str(y_pred_value[index]))
            plt.imshow(reconstructions[index], cmap="binary")
            plt.axis("off")
            
        plt.show()

if __name__ == "__main__":
    tf.app.run()