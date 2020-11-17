import time
import scipy as sp
import numpy as np
from numpy import ones, ones_like, zeros, zeros_like
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers import ActivityRegularization, BatchNormalization, Dropout
from keras.layers.merge import Maximum, Minimum, Concatenate, Multiply
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adagrad
from keras import regularizers
import keras.backend as K
from keras.losses import binary_crossentropy
# from keras.losses import huber_loss
from keras.constraints import Constraint
from utilities import binarise, batch, rand_batch
from utilities import output_progress, plot_sample, plot_TPR_metrics
from utilities import plot_confusion_matrix
import globals
from IPython.core.debugger import set_trace

GAN_DIR = globals.GAN_DIR
ADV_DIR = globals.ADV_DIR

plt.rc('text', usetex=globals.usetex)
plt.rc('font', family='serif')


class EvadeGAN:

    def __init__(self, target_model, x_dim=10000, z_dim=100, g_input='xz',
                 g_params={}, d_params={}, d_compile_params={},
                 gan_compile_params={}, summary=False, bin_threshold=0.5):
        self.graph = tf.compat.v1.get_default_graph()
        self.target_model = self.TargetModel(target_model)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.bin_threshold = bin_threshold
        self.g_input = g_input
        if self.g_input == 'z':
            self.name = 'EvadeGANz'
            self.setting = "Sample-Independent Perturbations Z"
            self.save_dir = 'EvadeGANz/'
        elif self.g_input == 'x':
            self.name = 'EvadeGANx'
            self.setting = "Sample-Dependent Perturbations X"
            self.save_dir = 'EvadeGANx/'
        else:
            self.name = 'EvadeGANxz'
            self.setting = "Sample-Dependent Perturbations XZ"
            self.save_dir = 'EvadeGANxz/'

        if summary:
            print(f"Summary of {self.name} Models [{self.setting}]:\n"
                  + '=' * 62 + '\n')

        # Build the generator
        self.generator = self.build_generator(**g_params, summary=summary)

        # Build & compile the discriminator
        self.discriminator = self.build_discriminator(**d_params,
                                                      **d_compile_params,
                                                      summary=summary)

        # Build & compile the adversarial network, GAN
        self.GAN = self.build_GAN(**gan_compile_params, summary=summary)

        # Combine logs
        self.log_params = {'G': [self.g_log],
                           'D': [self.d_log],
                           'GAN': [self.gan_log]}

    def build_generator(self, n_hidden=256, h_activation='relu',
                        regularizers={}, batchnorm=False,
                        out_activation='sigmoid',
                        drop_rate=0, summary=False):
        """Builds a generator using the passed hyperparameters"""

        # Input: xz, z, or x
        x = Input(shape=(self.x_dim,), name='g_x_input')
        z = Input(shape=(self.z_dim,), name='g_z_input')
        if self.g_input == 'z':
            g_input = z
        elif self.g_input == 'x':
            g_input = x
        else:
            g_input = Concatenate(axis=1, name='g_xz_input')([x, z])

        # Hidden
        hidden = Dense(n_hidden,
                       activation=h_activation,
                       name='g_hidden_relu')(g_input)

        if batchnorm:
            hidden = BatchNormalization(name='g_hidden_bn')(hidden)

        perturb = Dense(self.x_dim,
                        activation=out_activation,
                        **regularizers,
                        name='g_perturb_sigmoid')(hidden)

        # Dropout
        perturb = Dropout(drop_rate, name='perturb_dropout')(perturb)
        perturb = K.minimum(perturb,
                            1)  # NB: dropout scales up the kept inputs,
        # so clip to stay <=1 (for max later)..
        # use K.clip Or K.minimum(perturb, 1)
        # Output
        x_adv = Maximum(name='g_adv_max')([perturb, x])

        self.generator = Model([x, z], x_adv, name='Generator')
        if summary:
            self.generator.summary();
            print()

        # G parameters for logging
        self.reg = get_reg_factors(regularizers)
        self.g_log = {'in': self.g_input,
                      'h': f'({n_hidden},{h_activation})', 'bn': batchnorm,
                      'reg': self.reg, 'drop': drop_rate}

        return self.generator

    def build_discriminator(self, n_hidden=256, h_activation=None,
                            h_constraint=None, out_activation='sigmoid',
                            summary=False,
                            loss='binary_crossentropy', metrics=['accuracy'],
                            optimizer=Nadam(lr=0.001, clipvalue=1.0)):
        """Builds a discriminator using the passed hyperparameters"""
        x = Input(shape=(self.x_dim,), name='d_x_input')
        hidden = Dense(n_hidden, activation=h_activation,
                       kernel_constraint=h_constraint, name='d_hidden_linear')(
            x)
        pred = Dense(1, activation=out_activation, name='d_pred')(hidden)
        self.discriminator = Model(x, pred, name="Disriminator")
        if summary:
            self.discriminator.summary();
            print()

        self.discriminator.compile(loss=loss,
                                   optimizer=optimizer,
                                   metrics=metrics)

        self.discriminator.trainable = False

        # D parameters, for logging
        self.d_log = {"loss": 'bce' if loss == 'binary_crossentropy' else loss,
                      "opt": {type(optimizer).__name__:
                                  (optimizer.lr.numpy(), optimizer.clipvalue)}}

        return self.discriminator

    def build_GAN(self, loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Nadam(lr=0.001, clipvalue=1.0), beta=1.0,
                  normalise_loss=False, target_label=0,
                  bound_func='mean', max_changes=12, summary=False):
        """Builds an adversarial netowrk GAN using the passed hyperparameters"""

        x = Input(shape=(self.x_dim,), name='gan_x_input')
        z = Input(shape=(self.z_dim,), name='gan_z_input')
        x_adv = self.generator([x, z])

        self.discriminator.trainable = False

        y_pred = self.discriminator(x_adv)  # predictions

        self.GAN = Model([x, z], y_pred, name='GAN')

        if summary:
            self.GAN.summary();
            print()

        # Binarise to get a valid sample
        x_adv_bin = binarise(x_adv, self.bin_threshold)

        # Optional: Minimise the score of the target model (Add to loss)
        # Target label (goodware)
        # y_target = target_label * ones(x_adv.get_shape().as_list()[0])
        # loss_target_model = self.target_model.score(x_adv_bin, y_target)
        # self.GAN.add_loss(loss_target_mode)

        # Reduction function for the bound loss: mean or max (more restrictive)
        reduce_func = tf.reduce_max if bound_func == 'max' else tf.reduce_mean

        # Whether to scale the bound loss to the range [0, 1]
        scale = 1 / self.x_dim if normalise_loss else 1.0
        loss_bound = \
            reduce_func(
                tf.maximum(0.0,  # OR tf.zeros((tf.shape(x_adv)[0])),
                           tf.norm((x_adv_bin - x), ord=1,
                                   axis=1) - max_changes) * scale)

        # combined_loss = alpha*loss_target_model + beta*loss_changes

        self.GAN.add_loss(beta * loss_bound)

        self.GAN.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=metrics)

        self.gan_log = {
            "loss": f'custom({bound_func}, {max_changes}, {beta})',
            "opt": {type(optimizer).__name__:
                        (optimizer.lr.numpy(), optimizer.clipvalue)}}

        return self.GAN

    class TargetModel:
        def __init__(self, model):
            # Get model parameters (weights & intercept)
            if type(model) == LinearSVC:
                w = model.coef_.flatten()
            elif type(model) == SVC:
                w = model.coef_.toarray().flatten()

            b = model.intercept_[0]

            self.weights = tf.Variable([w], dtype=tf.float32, trainable=False)
            self.intercept = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.classes = tf.Variable(model.classes_, trainable=False)
            self.accuracy = tf.keras.metrics.BinaryAccuracy()

        def predict(self, X):
            # Get decision function
            # X = tf.convert_to_tensor(X, dtype=tf.float32) # in case no tensor
            scores = K.dot(X, tf.transpose(self.weights)) + self.intercept

            # Classify
            idx = tf.cast(scores > 0, tf.int32)
            # y_pred = K.get_value(self.classes)[idx]     # Class Label, eager
            y_pred = tf.gather(self.classes, idx)
            return y_pred

        def score(self, X, y_target):
            y_pred = self.predict(X)
            self.accuracy.update_state(y_pred, y_target)
            return self.accuracy.result()

    def train(self, target_model, X_mal_train, X_mal_test, X_good_train,
              X_good_test, mal_label=1, good_label=0, earlystop=False,
              zmin=0, zmax=1, epochs=500, batch_size=32, combined_d_batch=False,
              d_train_mal=False, d_train_adv=True, good_batch_factor=1,
              d_times=1, gan_times=1, n_progress=1, minTPR_threshold=0,
              max_changes=np.inf, gan_dir=GAN_DIR, smooth_alpha=1.0,
              sample_train=True):
        """
        Performs GAN training.
        :param target_model: The target model of the evasion attack
        :param X_mal_train: The malware training set
        :param X_mal_test: The malware test set
        :param X_good_train: The goodware training set
        :param X_good_test: The goodware test set
        :param mal_label: The label for the malware class (original label)
        :param good_label: The label for the goodware class (target label)
        :param zmin: The lower bound of the random noise
        :param zmax: The upper bound of the random noise
        :param epochs: The number of training epochs
        :param batch_size: The size of a training batch
        :param d_train_mal: Whether to train the disciminator on malware.
        :param combined_d_batch: Whether to train the discriminator on one batch
                  that combine all classes or train on each eparately
        :param good_batch_factor: The size ratio of a goodware batch compared
                  to that of a malware batch.
        :param d_times: The number of times to train the discriminator in each
                  iteration.
        :param gan_times: The number of times to train the GAN in each iteration
        :param n_progress: The number of epochs with no improvement/output after
                  which print ouput to check for progress.
        :param minTPR_threshold: The threshold to which we wish to minimise the
                  the True Positive Rate (TPR).
        :param max_changes: A constraint on the maximum number of changes in
                  generated adversarial examples (AEs)
        :return: tuple (
                    TPR_train: The list of TPR scores on the training set at
                                each epoch,
                    TPR_test: The list of TPR scores on the test set at each
                              epoch,
                    avg_diff_train: The list of avg changes in AEs generated
                              from training set at each epoch,
                    avg_diff_test: The list of avg changes in AEs generated
                              from the test set at each epoch,
                    d_metrics: The list of the discriminator metrics
                              [loss, accuracy] at each epoch,
                    gan_metrics: The list of the GAN metrics
                              [loss, accuracy] at each epoch,
                    best_G_path: The path to the best performing G model
                  )
        """

        g_batch_size = good_batch_factor * batch_size

        # Metrics accumulators
        d_metrics = []
        gan_metrics = []

        # Initial TPR on the training & test sets
        TPR_train = [target_model.score(X_mal_train,
                                        mal_label * ones(X_mal_train.shape[0]))]
        TPR_test = [target_model.score(X_mal_test,
                                       mal_label * ones(X_mal_test.shape[0]))]
        minTPR = 1.0
        minTPR_avg_changes = -1
        minTPR_max_changes = -1
        min_epoch = output_epoch = 0
        best_G_path = None

        print(f"Initial TPR on the training set: {TPR_train}")
        print(f"Initial TPR on the test set: {TPR_test}\n")

        # Average changes (perturbations) in adversarial examples
        avg_diff_train = []
        avg_diff_test = []

        # IDs for plots
        plot_id = 1
        gan_id = 1
        tpr_id = 1

        t1 = time.perf_counter()

        for epoch in range(epochs):
            # Generate batches of size (gan_times * batch_size)
            X_mal_batches = batch(X_mal_train, gan_times * batch_size,
                                  seed=epoch)
            # Epoch metrics accumulators
            d_metrics_epoch = np.empty((0, 2))
            gan_metrics_epoch = np.empty((0, 2))

            for X_mal_batch in X_mal_batches:
                ################################################################
                # Train the discriminator for d_times iterations
                ################################################################
                # Generate minibatches of size batch_size
                minibatches = batch(X_mal_batch, batch_size, seed=epoch)
                d_metrics_batch = np.empty((0, 2))
                # Train for d_times
                for i in range(d_times):
                    # __could reseed with (epoch + i) for reproducibility__
                    X_mal = next(minibatches, None)  # Use these batches first
                    if X_mal is None:  # Then generate randomly
                        X_mal = rand_batch(X_mal_train, batch_size)

                    Y_mal = smooth_alpha * mal_label * ones(
                        X_mal.shape[0])  # Smooth

                    noise = np.random.uniform(zmin, zmax,
                                              size=[batch_size, self.z_dim])

                    # Generate adversarial examples
                    X_adv = self.generator.predict([X_mal, noise])
                    X_adv = binarise(X_adv, self.bin_threshold)
                    Y_adv = target_model.predict(X_adv)
                    Y_adv[
                        Y_adv == mal_label] = smooth_alpha * mal_label  # Smooth

                    X_good = rand_batch(X_good_train, g_batch_size)
                    Y_good = good_label * ones(X_good.shape[0])     # Good_Label

                    # Train the discriminator
                    self.discriminator.trainable = True

                    if combined_d_batch:
                        # *** Train once on a combined batch ****
                        X = X_good
                        Y = Y_good
                        if d_train_mal:
                            X = np.concatenate((X, X_mal))
                            Y = np.concatenate((Y, Y_mal))
                        if d_train_adv:
                            X = np.concatenate((X, X_adv))
                            Y = np.concatenate((Y, Y_adv))
                        metrics = self.discriminator.train_on_batch(X, Y)
                    else:
                        # ** Train on separate batches & combine metrics **
                        metrics_good = self.discriminator.train_on_batch(X_good,
                                                                         Y_good)
                        metrics_mal = self.discriminator.train_on_batch(X_mal,
                                                                        Y_mal) \
                            if d_train_mal else [np.nan, np.nan]
                        metrics_adv = self.discriminator.train_on_batch(X_adv,
                                                                        Y_adv) \
                            if d_train_adv else [np.nan, np.nan]
                        # Avg metrics
                        metrics = np.nanmean(np.array([metrics_mal,
                                                       metrics_good,
                                                       metrics_adv]), axis=0)

                    # Accumulate metrics for d_times iterations
                    d_metrics_batch = np.vstack((d_metrics_batch, metrics))

                # Average the metrics of all d_times iterations
                d_metrics_batch = np.mean(d_metrics_batch, axis=0)
                # Add to discriminator metrics for this epoch
                d_metrics_epoch = np.vstack((d_metrics_epoch, metrics))

                ################################################################
                # Train the Generator
                ################################################################
                # Generate minibatches of size batch_size
                minibatches = batch(X_mal_batch, batch_size, seed=epoch)
                gan_metrics_batch = np.empty((0, 2))
                # Train for gan_times
                for i in range(gan_times):
                    # Number of minibatches should be exactly gan_times
                    X_mal = next(minibatches, None)
                    if X_mal is None:  # Just in case, generate randomly
                        X_mal = rand_batch(X_mal_train, batch_size)

                    noise = np.random.uniform(zmin, zmax, size=[batch_size,
                                                                self.z_dim])
                    self.discriminator.trainable = False

                    # Train with target label = GOOD_LABEL
                    metrics = self.GAN.train_on_batch([X_mal, noise],  # <<<<
                                                      good_label * ones(
                                                          X_mal.shape[0]))
                    # discriminator.trainable = True

                    # Accumulate metrics for gan_times iterations
                    gan_metrics_batch = np.vstack((gan_metrics_batch, metrics))

                # Average the metrics of all gan_times iterations
                gan_metrics_batch = np.mean(gan_metrics_batch, axis=0)
                # Add to the generator metrics for this epoch
                gan_metrics_epoch = np.vstack((gan_metrics_epoch, metrics))

            # Average metrics of each epoch
            d_metrics.append(np.mean(d_metrics_epoch, axis=0).tolist())
            gan_metrics.append(np.mean(gan_metrics_epoch, axis=0).tolist())
            gan_loss = gan_metrics[-1][0]

            # TPR on adversarial training set
            noise = np.random.uniform(zmin, zmax, (X_mal_train.shape[0],
                                                   self.z_dim))
            X_adv_train = binarise(self.generator.predict([X_mal_train, noise]),
                                   self.bin_threshold)
            # Score with target label = MAL_LABEL
            Y_adv_train = mal_label * ones(X_adv_train.shape[0])  # MAL_LABEL
            TPR = target_model.score(X_adv_train, Y_adv_train)
            TPR_train.append(TPR)

            # Changes (L1 norms) in the adversarial training set
            diff_train = norm((X_adv_train - X_mal_train), ord=1, axis=1)
            avg_diff_train_current = np.mean(diff_train)
            max_diff_train_current = np.max(diff_train)
            avg_diff_train.append(avg_diff_train_current)

            # TPR on adversarial test set
            noise = np.random.uniform(zmin, zmax, (X_mal_test.shape[0],
                                                   self.z_dim))

            X_adv_test = binarise(self.generator.predict([X_mal_test, noise]),
                                  self.bin_threshold)
            Y_adv_test = mal_label * ones(X_adv_test.shape[0])  # MAL_LABEL
            TPR = target_model.score(X_adv_test, Y_adv_test)
            TPR_test.append(TPR)

            # Changes (L1 norms) in the adversarial test set
            diff_test = norm((X_adv_test - X_mal_test), ord=1, axis=1)
            avg_diff_test_current = np.mean(diff_test)
            max_diff_test_current = np.max(diff_test)
            avg_diff_test.append(avg_diff_test_current)

            # Output progress if TPR has decreased (improved evasion)
            # ... or if TPR is the same but avg changes have decreased
            if (TPR < minTPR) or \
                (TPR == minTPR and avg_diff_test_current < minTPR_avg_changes):  # check avg or max
                print("\n>>>> New Best Results: "
                      f"Previous minTPR: [{minTPR:.8f}] ==> "
                      f"New minTPR: [{TPR:0.8f}] "
                      f"GAN Loss: [{gan_loss:.8f}]  <<<<")
                output_progress(epoch, TPR_train, TPR_test,
                                diff_train, diff_test)
                minTPR = TPR
                min_epoch = output_epoch = epoch
                minTPR_avg_changes = avg_diff_test_current
                minTPR_max_changes = max_diff_test_current
                minTPR_std = np.std(diff_test)
                minTPR_quantiles = np.quantile(diff_test, [0.25, 0.5, 0.75])

                # Save weights
                minTPR_weights_path = \
                    (gan_dir + self.save_dir + 'weights/' +
                     f'GAN_minTPR_weights_epoch_{epoch}_'
                     f'TPR_{minTPR:.2f}_dtimes_{d_times}_changes_'
                     f'{avg_diff_test_current:.0f}_actReg_{self.reg[0]}_'
                     + time.strftime("%m-%d_%H-%M-%S") + '.h5')
                self.GAN.save_weights(minTPR_weights_path)

                # Generate and plot a sample of AEs
                sample_sz = 10
                sample_noise = np.random.uniform(zmin, zmax, size=[sample_sz,
                                                                   self.z_dim])

                if sample_train:  # Sample from training
                    sample_mal = rand_batch(X_mal_batch, sample_sz)
                else:  # Sample from test set
                    sample_mal = np.asarray(rand_batch(X_mal_test, sample_sz))

                plot_sample(sample_mal, sample_noise, self.generator,
                            target_model, epoch, TPR_train=TPR_train,
                            TPR_test=TPR_test, params=self.log_params,
                            avg_changes=avg_diff_test_current,
                            m_label=mal_label, g_label=good_label,
                            annotate=False, out_dir=ADV_DIR, plot_id=plot_id)
                plot_id = plot_id + 1

                if minTPR <= minTPR_threshold:
                    print(
                        "\n" + "#" * 150 + "\n"
                        f"# Target Evasion Rate {100 * (1 - TPR):.2f}% "
                        f"achieved at epoch [{epoch}], "
                        f"with avg {avg_diff_test_current:.1f} "
                        f"& max {max_diff_test_current:.1f} changes per sample "
                        f"(on the test set) ... "
                        f"GAN Loss: [{gan_loss:.8f}]"
                        "\n" + "#" * 150 + "\n"
                    )

                    if minTPR_avg_changes <= max_changes:
                        print("Training CONVERGED. "
                            "Target Evasion Rate achieved within max changes..."
                            "TRAINING ENDS HERE #")
                        # Save generator
                        best_G_path = \
                            (gan_dir + self.save_dir + 'models/' +
                            f'G_Target_TPR_epoch_{epoch}_'
                            f'TPR_{minTPR:.2f}_dtimes_{d_times}_changes_'
                            f'{avg_diff_test_current:.0f}_actReg_{self.reg[0]}_'
                            + time.strftime("%m-%d_%H-%M-%S") + '.h5')
                        self.generator.save(best_G_path)

                        if earlystop:
                            break

            # If no better than minTPR, but still achieved target evasion, ...
            elif TPR <= minTPR_threshold:
                # output_epoch = epoch
                print(
                    "\n" + "#" * 150 + "\n"
                    f"# Target Evasion Rate {100 * (1 - TPR):.2f}% "
                    f"achieved at epoch [{epoch}] "
                    f"with avg {avg_diff_test_current:.1f} "
                    f"and max {max_diff_test_current:.1f} changes per sample "
                    f"(on the test set) ... "
                    f"GAN Loss: [{gan_loss:.8f}]"
                    "\n" + "#" * 150 + "\n"
                )

                # Save weights
                weights_path = \
                    (gan_dir + self.save_dir + 'weights/' +
                     f'GAN_minTPR_weights_epoch_{epoch}_'
                     f'TPR_{minTPR:.2f}_dtimes_{d_times}_changes_'
                     f'{avg_diff_test_current:.0f}_actReg_{self.reg[0]}_'
                     + time.strftime("%m-%d_%H-%M-%S") + '.h5')
                # self.GAN.save_weights(file_path)

                # If within max changes
                if avg_diff_test_current <= max_changes:  # check avg or max?
                    print("Target Evasion Rate achieved within max changes...")
                    # Save model
                    model_path = \
                        (gan_dir + self.save_dir + 'models/' +
                        f'GAN_Target_TPR_epoch_{epoch}_'
                        f'TPR_{minTPR:.2f}_dtimes_{d_times}_changes_'
                        f'{avg_diff_test_current:.0f}_actReg_{self.reg[0]}_'
                        + time.strftime("%m-%d_%H-%M-%S") + '.h5')
                    # self.GAN.save(model_path)
                    if earlystop:
                        break
                else:
                    print()
                    # Maybe adjust weights
                    # print("Should we adjust regulizers?")
                    # generator.layers[-2].rate *= 0.1
                    # generator.layers[-3].activity_regularizer.l1 *= 0.1
                    # generator.layers[-3].activity_regularizer.l2 *= 0.1
                    # weights = generator.get_weights()
                    # generator = keras.models.clone_model(generator)
                    # generator.set_weights(weights)
                    # Adapt regularisation weights
                    # K.set_value(l1_factor, 0.1*l1_factor)
                    # K.set_value(l2_factor, 0.1*l2_factor)

            if (epoch + 1 - output_epoch) > n_progress:
                # If no new imporovement for for a while, output progress
                output_epoch = epoch
                print(f"\n*** Checking progress *** "
                      f"GAN Loss: [{gan_loss:.8f}] ***")
                output_progress(epoch, TPR_train, TPR_test,
                                diff_train, diff_test)

                # Generate and plot a sample of AEs
                sample_sz = 10
                sample_noise = np.random.uniform(zmin, zmax, size=[sample_sz,
                                                                   self.z_dim])

                sample_mal = rand_batch(X_mal_batch, sample_sz)

                plot_sample(sample_mal, sample_noise, self.generator,
                            target_model, epoch, TPR_train=TPR_train,
                            TPR_test=TPR_test, params=self.log_params,
                            avg_changes=avg_diff_test_current,
                            m_label=mal_label, g_label=good_label,
                            annotate=False, out_dir=ADV_DIR, plot_id=plot_id)
                plot_id = plot_id + 1

        t2 = time.perf_counter()
        print("\n\n" + "#" * 165 + "\n"
            f"# Finished {epoch + 1} epochs in {(t2 - t1) / 60:.2f} minutes\n"
            f"# Best Evastion Rate = {100 * (1 - minTPR):.4f}% "
            f"(lowest TPR = {100 * minTPR:.4f}%) "
            f"achieved after {min_epoch + 1} epochs, with avg "
            f"{minTPR_avg_changes:.1f} \u00b1 SD({minTPR_std:.1f}) | "
            f" Q1-3  {minTPR_quantiles.astype(int).tolist()} | "
            f" and max {minTPR_max_changes:.1f} "
            f"changes per sample.\n"
            + "#" * 165 + "\n\n")

        return TPR_train, TPR_test, \
               avg_diff_train, avg_diff_test, \
               d_metrics, gan_metrics, \
               best_G_path


def my_regularizer(x, l1_weight=0, l2_weight=0, bin_threshold=0.5):
    return l1_weight * tf.reduce_sum(tf.abs(binarise(x, bin_threshold))) + \
           l2_weight * tf.reduce_sum(tf.square(x))


def bound_loss(x_adv, x_mal, method='l2mean', weight=0.001, max_changes=12,
               bin_threshold=0.5):
    x_adv = binarise(x_adv, bin_threshold)
    n_changes = \
        tf.reduce_mean(tf.norm((x_adv - x_mal), ord=1, axis=1) - max_changes)
    return weight * n_changes


def get_reg_factors(regularizers):
    """Returns a list of L1_L2 regularization factors, for logging purposes"""
    return [
        None if v is None else (round(v.l1.item(), 5), round(v.l2.item(), 5))
        for _, v in sorted(regularizers.items())]


