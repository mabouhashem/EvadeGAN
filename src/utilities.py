import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import keras.backend as K
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.sparse import SparseTensor  # Requires tensorflow 2.2
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import globals

PLOT_DIR = globals.PLOT_DIR

plt.rc('text', usetex=globals.usetex)
plt.rc('font', family='serif')

def get_Y(X, label):
    n = X.shape[0]
    if label == 0:
        return np.zeros(n)
    elif label == 1:
        return np.ones(n)
    else:
        return label * np.ones(n)


def toSparseTensor(X):
    X = X.tocoo()
    indices = np.vstack((X.row, X.col)).T  # Transpose
    values = X.data
    shape = X.shape

    return SparseTensor(indices, values, shape)


def cmap(pos='Reds', neg='Greens_r', n=32, n_mid=20):
    """Generates a custom colormap"""
    pos = cm.get_cmap(pos, n)
    neg = cm.get_cmap(neg, n)
    mid = np.ones((n_mid, 4))

    colors = np.vstack((neg(np.linspace(0, 1, n)),
                        mid,
                        pos(np.linspace(0, 1, n))))

    colormap = ListedColormap(colors, name='GnWtRd')

    return colormap

    # GnWtRd = LinearSegmentedColormap.from_list('GnWtRd',
    #                                            ['g', 'g', 'g', 'w', 'w', 'r',
    #                                             'r', 'r'], 50)
    # RdGn = sns.diverging_palette(h_neg=150, h_pos=10, s=90, l=80, sep=3,
    #                              center='light', as_cmap=True)


###############################################################################
# Attack Utils
###############################################################################

def binarise(X, threshold=0.5):
    return tf.where(X > threshold, 1.0, 0.0)


def batch(X, batch_size, seed=0, iterator=True):
    """"
    Partitions a dataset into batches, returning a batch dataset or an iterator.
    :param X: The dataset to batch
    :param batch_size: The size of each batch
    :param seed: The shuffle seed
    :retrun: A tensor batch dataset, or as a numpy iterator.
    """
    # buffer_size = int(1e6)
    buffer_size = X.shape[0]  # For perfect shuffle, buff is the size of X

    if K.is_sparse(X):  # If a sparse tensor
        X = K.to_dense(X)
    elif sp.sparse.issparse(X):  # If a sparse matrix
        X = X.todense()

    batches = Dataset.from_tensor_slices(X). \
        shuffle(buffer_size=buffer_size, seed=seed). \
        batch(batch_size, drop_remainder=True)

    if iterator:
        batches = batches.as_numpy_iterator()

    return batches


def rand_batch(X, batch_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return X[np.random.choice(X.shape[0], batch_size, replace=False)]


def output_progress(epoch, TPR_train, TPR_test, diff_train, diff_test):
    """
    Outputs the porgress of training at a given epoch.
    """
    # Performance on the training set
    diff_avg = np.mean(diff_train)
    diff_std = np.std(diff_train)
    diff_quantiles = np.quantile(diff_train, [0.25, 0.5, 0.75])
    diff_max = np.max(diff_train)
    print(f"Epoch: {epoch}\n"
          f"{'TPR Training':<13}: {TPR_train[-1]:.8f} | "
          f"Avg # Changes: {diff_avg:.0f} \u00b1 SD({diff_std:.2f}) | "
          f" Change Quantiles (25%, 50%, 75%): "
          f"{diff_quantiles.astype(int).tolist()} | "
          f"Max {diff_max:.0f}")

    # Performance on the testing set
    diff_avg = np.mean(diff_test)
    diff_std = np.std(diff_test)
    diff_quantiles = np.quantile(diff_test, [0.25, 0.5, 0.75])
    diff_max = np.max(diff_test)
    print(f"{'TPR Testing':<13}: {TPR_test[-1]:.8f} | "
          f"Avg # Changes: {diff_avg:.0f} \u00b1 SD({diff_std:.2f}) | "
          f" Change Quantiles (25%, 50%, 75%): "
          f"{diff_quantiles.astype(int).tolist()} | "
          f"Max: {diff_max:.0f}\n")


def plot_sample(X_mal, noise, generator, target_model, epoch,
                TPR_train, TPR_test, avg_changes, m_label=1, g_label=0,
                params={}, annotate=True, out_dir='.', plot_id=None,
                xz_input=True, dpi=100):
    """
    Generates and plots adversarial examples for a random sample of malware.
    """
    sample_sz = 8
    if type(target_model) == LinearSVC:
        weights = target_model.coef_.flatten()
    elif type(target_model) == SVC:
        weights = target_model.coef_.toarray().flatten()

    # Top N negative features
    N = 20
    top_neg = np.argpartition(weights, range(N))[:N]
    top_neg = np.unravel_index(top_neg, (100, 100))
    top_neg = list(zip(*top_neg))
    weights = np.round(weights.reshape((100, 100)), 2)

    Y_mal = target_model.predict(X_mal)
    DF_mal = target_model.decision_function(X_mal)

    X_adv = generator.predict([X_mal, noise])
    X_adv = binarise(X_adv).numpy()  # numpy to reshape
    Y_adv = target_model.predict(X_adv)
    DF_adv = target_model.decision_function(X_adv)

    n_feats_mal = np.count_nonzero(X_mal, axis=1)
    diff = X_adv - X_mal
    dist1 = np.linalg.norm(diff, ord=1, axis=1)
    dist2 = np.linalg.norm(diff, ord=2, axis=1)

    fig = plt.figure(num='Sample', figsize=(16, 16), facecolor='w', dpi=dpi)
    title = \
        (f"A sample of original malware \& corresponding AEs "
         f"[Epoch: {epoch + 1}] "
         f"[Evasion Rates (Current, Best)\%: "
         f"Test: ({100 * (1 - TPR_test[-1]):.2f}, {100 * (1 - min(TPR_test)):.2f}) "
         f"Train: ({100 * (1 - TPR_train[-1]):.2f}, {100 * (1 - min(TPR_train)):.2f})] "
         f"[Avg \# changes: {avg_changes:.0f}]")
    fig.suptitle(title, c='r', x=0.5, y=0.99, fontsize=16, fontweight='bold',
                 bbox=dict(facecolor='none', edgecolor='red'))
    font_param = {'size': 14, 'color': 'k'}
    fig.text(0.5, 0.95, params, fontsize=14, ha='center', va='center',
             bbox=dict(fc='none', ec='k', pad=6))
    idx = 0
    for row in range(1, 2 * sample_sz + 1, sample_sz):
        for i in range(sample_sz // 2):
            subplot = row + i
            # Malware subplots
            ax = plt.subplot(4, 4, subplot)
            img = X_mal[idx].reshape((100, 100))
            plt.imshow(img, cmap='gray', interpolation='none')
            # pred_color = {M (mal): k, G (good): r}
            pred, color = ('M', 'k') if Y_mal[idx] == m_label else ('G', 'r')
            pred = r"$\bf{[" + pred + "]}$"  # Predicted label in bold
            ax.set_title(f'{pred} DF({DF_mal[idx]:.2f})  ::  '
                         f'Features({n_feats_mal[idx]:.0f})',
                         color=color, fontsize=14)
            plt.axis('off')

            # Adversarial subplots
            ax = plt.subplot(4, 4, subplot + 4)
            img = X_adv[idx].reshape((100, 100))
            plt.imshow(img, cmap='gray', interpolation='none')
            # pred_color = {M (mal): r, G (good): g}
            pred, color = ('M', 'r') if Y_adv[idx] == m_label else ('G', 'g')
            pred = r"$\bf{[" + pred + "]}$"  # Predicted label in bold
            img = diff[idx].reshape((100, 100))
            # plt.imshow(img, alpha=img, cmap='spring_r')

            # Plot added features
            y, x = np.where(img == 1)
            c = []
            for i, j in zip(x, y):
                w = weights[j][i]
                if w < 0:  # Features with -ve weights
                    c.append('g')
                elif w > 0:
                    c.append('r')  # Features with +ve weights
                else:
                    c.append('c')

            plt.scatter(x, y, s=25, marker='o', c=c)
            # plt.scatter(x, y, s=100, marker='o', c='None', ec='y')  # frame
            # Annotate with weights
            if annotate or dist1[idx] <= 15:  # Annotate if <12
                for i, j in zip(x, y):
                    w = weights[j][i]
                    if w != 0:  # Annotate non-zero w
                        c = 'yellow' if w < 0 else 'darkorange'
                        fw = 'bold' if (j,
                                        i) in top_neg else 'normal'  # Top feat
                        ax.annotate(w, (i, j), (i, j - 1), size=10, c=c,
                                    weight=fw)

            ax.set_title(f'{pred} DF({DF_adv[idx]:.2f})  ::  '
                         f'L1({dist1[idx]:.0f})  ::  '
                         f'L2({dist2[idx]:.1f})',
                         color=color, fontsize=14)
            plt.axis('off')
            idx = idx + 1

    # Row labels
    font_mal = {'size': 16, 'weight': 'bold', 'color': 'k'}
    font_adv = {'size': 16, 'weight': 'bold', 'color': 'r'}
    fig.text(0.002, 0.82, 'Original', va='top', rotation='vertical',
             fontdict=font_mal)
    fig.text(0.002, 0.57, 'Adversarial', va='center', rotation='vertical',
             fontdict=font_adv)
    fig.text(0.002, 0.35, 'Original', va='center', rotation='vertical',
             fontdict=font_mal)
    fig.text(0.002, 0.12, 'Adversarial', va='center', rotation='vertical',
             fontdict=font_adv)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(out_dir + f'{id}_epoch_{epoch}_loss_{loss}.png')
    plt.show()


def plot_TPR_metrics(TPR_train, TPR_test, avg_diff_train, avg_diff_test,
                     d_metrics, gan_metrics, plot_id=0, titles=True, dpi=600):
    """
    Plots the TPR of the target model during training, as well as the metrics
    of the GAN and the discriminator.
    :param TPR_train: The TPR of the target model on the taining set
    :param TPR_test: The TPR of the target model on the taining set
    :param d_metrics: The training metrics of the discriminator (loss, accuracy)
    :param gan_metric: The training metrics of the GAN
    :plot_id: A unique ID for the plot, for logging
    :return: None
    """

    fig = plt.figure(num='TPR_Metrics', figsize=(20, 6), facecolor='w', dpi=dpi)

    # 1.a Plot the TPR of the target model
    epochs = len(TPR_train)
    minTPR = min(TPR_test)
    min_idx = TPR_test.index(minTPR)
    ax1 = plt.subplot(1, 3, 1)
    if titles:
        ax1.set_title('TPR of the Target Model \& Average \# Changes per AE',
                      fontsize=16, fontweight='bold')
    ax1.vlines(1, ymin=0, ymax=1, linestyles='dashed', linewidth=1)    # Initial
    # plt.scatter(min_idx, minTPR, s=200, marker='o', c='None', ec='r')# Minimum
    # ax1.vlines(min_idx, ymin=0, ymax=1, linewidth=3, color='k')      # Minimum
    # ax1.fill_between([0, 1], -1, 1)
    ax1.plot(range(epochs), TPR_train, c='darkred', linestyle='-',
             label='Training TPR', linewidth=2)
    ax1.plot(range(epochs), TPR_test, c='limegreen', linestyle='--',
             label='Test TPR', linewidth=2)
    ax1.set_ylabel('TPR', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc='upper left', bbox_to_anchor=(0.06, 1.))

    # 1.b Plot the avg # changes per AE
    ax1b = ax1.twinx()
    ax1b.plot(range(1, epochs), avg_diff_train, c='mediumblue',
              label='Training Set Changes', linewidth=2)
    ax1b.plot(range(1, epochs), avg_diff_test, c='magenta', linestyle='--',
              label='Test Set Changes', linewidth=2)
    ax1b.set_ylabel('Changes (L1 Distance)', fontsize=14)
    ax1b.set_ylim(0, int(max(max(avg_diff_train), max(avg_diff_test))) + 1)
    ax1b.legend(loc='upper right')

    # 2. Plot the metrics (loss & accuracy) of the GAN and the discriminator
    d_metrics = np.array(d_metrics)
    gan_metrics = np.array(gan_metrics)

    ax2 = plt.subplot(1, 3, 2)
    if titles:
        ax2.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax2.plot(range(1, epochs), gan_metrics[:, 0], c='g',
             label='GAN', linewidth=2)
    ax2.plot(range(1, epochs), d_metrics[:, 0], c='r',
             label='Discriminator', linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    if titles:
        ax3.set_title('Training Accuracy', fontsize=16, fontweight='bold')
    ax3.plot(range(1, epochs), gan_metrics[:, 1], c='g',
             label='GAN', linewidth=2)
    ax3.plot(range(1, epochs), d_metrics[:, 1], c='r',
             label='Discriminator', linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=14)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_ylabel("Accuracy", fontsize=14)
    ax3.legend()

    plt.tight_layout()

    # plt.savefig(TPR_DIR + f'TPR_{plot_id}.png')
    plt.show()


def plot_metrics(d_metrics, gan_metrics, plot_id=0):
    """
    Plots performance metrics (loss, accuracy) during GAN training.
    :param d_metrics: The training metrics of the discriminator (loss, accuracy)
    :param gan_metric: The training metrics of the GAN
    :plot_id: A unique ID for the plot, for logging
    :return: None
    """

    d_metrics = np.array(d_metrics)
    gan_metrics = np.array(gan_metrics)

    fig = plt.figure(num='Training Metrics', figsize=(16, 4), facecolor='w')

    # fig.suptitle("Training Metrics", c='r', fontsize=16, fontweight='bold',
    #           bbox=dict(facecolor='none', edgecolor='red'))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Training Loss', fontsize=18, fontweight='bold')
    plt.plot(range(len(gan_metrics)), gan_metrics[:, 0], c='g',
             label='GAN', linewidth=2)
    plt.plot(range(len(d_metrics)), d_metrics[:, 0], c='r',
             label='Discriminator', linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Training Accuracy', fontsize=18, fontweight='bold')
    plt.plot(range(len(gan_metrics)), gan_metrics[:, 1], c='g',
             label='GAN', linewidth=2)
    plt.plot(range(len(d_metrics)), d_metrics[:, 1], c='r',
             label='Discriminator', linewidth=2)
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    # plt.savefig(METRICS_DIR + f'loss_acc_{plot_id}.png')
    plt.show()


def plot_TPR(train_TPR, test_TPR, plot_id=0):
    """
    Plots TPR during GAN training.
    :param train_TPR: The TPR of the target model on the taining set
    :param test_TPR: The TPR of the target model on the taining set
    :plot_id: A unique ID for the plot, for logging
    :return: None
    """
    epochs = len(train_TPR)
    plt.figure()
    plt.title('TPR of the Target Model', fontsize=18, fontweight='bold')
    plt.plot(range(epochs), train_TPR, c='r',
             label='Training Set', linewidth=2)
    plt.plot(range(epochs), test_TPR, c='g', linestyle='--',
             label='Test Set', linewidth=2)
    plt.ylabel('TPR')
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, epochs + 1))
    plt.legend()
    # plt.savefig(TPR_DIR + f'TPR_{plot_id}.png')
    plt.show()


def plot_confusion_matrix(Y_true_train, Y_pred_train, Y_true_test, Y_pred_test,
                          title=None, savefig=False, dpi=300):
    # Confusion matrix plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(13, 5), dpi=dpi)
    if title is not None:
        f.suptitle(title, fontsize=22)
    f.text(0.4, 0.03, 'Predicted label', ha='left', fontsize=20,
           fontweight='bold')
    f.text(0.05, 0.5, 'True label', ha='left', va='center', fontsize=20,
           rotation='vertical', fontweight='bold')
    labels = ['Malware', 'Goodware']
    plots = [(1, "Training Set", Y_true_train, Y_pred_train, ax1, True, 'Reds'),
             (2, "Test Set", Y_true_test, Y_pred_test, ax2, False, 'Blues')]

    for i, title, ytrue, ypred, ax, ylabel, cmap in plots:
        cm = confusion_matrix(ytrue, ypred,
                              labels=[1, 0],
                              normalize='true')
        hmap = sns.heatmap(cm, ax=ax, annot=True, square=True, fmt='.2f',
                           cmap=cmap, annot_kws={"size": 18, "weight": "bold"},
                           cbar_kws={"shrink": 1.})
        hmap.collections[0].colorbar.ax.tick_params(labelsize=14)
        for _, spine in hmap.spines.items():
            spine.set_visible(True)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xticklabels(labels, fontsize=16)
        ax.set_yticklabels(labels, fontsize=16, va='center')
        ax.set(xlabel='', ylabel='')

    if savefig:
        plt.savefig(PLOT_DIR + f'confusion_matrix2_{dpi}.svg',
                    dpi=dpi, transparent=True)
    plt.show()