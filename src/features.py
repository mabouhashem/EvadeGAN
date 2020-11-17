import numpy as np
import pandas as pd
from sklearn import feature_selection
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import cmap
import globals

PLOT_DIR = globals.PLOT_DIR
FEAT_DIR = globals.FEAT_DIR

plt.rc('text', usetex=globals.usetex)
plt.rc('font', family='serif')


pd.set_option('display.max_columns', 10)


def get_features(feature_names, weights, X, X_malware, X_goodware, Y):
    """
    Builds a feature dataframe for analysis and reporting purposes.
    :param weights: Feature weights
    :param X: Dataset features
    :param X_malware:  Malware features
    :param X_goodware: Goodware features
    :param Y: Dataset labels
    :return: features (DataFrame)
    """
    n_features = len(feature_names)

    features = pd.DataFrame({'feature': feature_names, 'weight': weights})

    col_idx = X.tocsc().nonzero()[1]
    all_count = np.bincount(col_idx, minlength=n_features)
    features['all_count'] = all_count
    all_freq = all_count / X.shape[0]
    features['all_freq'] = all_freq

    # feature frequencies in malware
    col_idx = X_malware.tocsc().nonzero()[1]
    mal_count = np.bincount(col_idx, minlength=n_features)
    features['mal_count'] = mal_count
    mal_freq = mal_count / X_malware.shape[0]
    features['mal_freq'] = mal_freq

    # feature frequencies in goodware
    col_idx = X_goodware.tocsc().nonzero()[1]
    good_count = np.bincount(col_idx, minlength=n_features)
    features['good_count'] = good_count
    good_freq = good_count / X_goodware.shape[0]
    features['good_freq'] = good_freq

    chi2, pval = feature_selection.chi2(X, Y)
    features['chi2'] = chi2
    features['pval'] = pval

    # features.to_csv(DIR + 'features.csv')

    return features


def features_per_samples(X):
    """
    Features per samples.
    :param X: The set
    :return:
    """
    nnz = X.getnnz(axis=1)                      # number of features in each row
    return pd.DataFrame(X.getnnz(axis=1)).describe()


def weight_summary(features):
    """
    Outputs a summary of weight distribution.
    :param features:
    :return: None
    """
    weights = features[['weight']]
    summary = weights.describe()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    bg = '#ACDCF6'
    ax1.axis("off")
    tbl = ax1.table(cellText=np.round(summary.values, 4), loc='center',
            rowLabels=summary.index, colLabels=summary.columns,
            rowColours=['w', bg, bg, bg, 'w', 'w', 'w', bg],
            cellColours=[['w'], [bg], [bg], [bg], ['w'], ['w'], ['w'], [bg]])

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.25, 1.25)

    weights.plot(ax=ax2, kind='kde',
                    xlim=(weights.weight.min(), weights.weight.max()))

    plt.tight_layout()
    plt.show()


def weight_density(features):
    weights = features[['weight']]
    ax = plt.gca()
    weights.plot(ax=ax, kind='kde', legend=None, fontsize=12,
                 xlim=(weights.weight.min(), weights.weight.max()))
    font = {'size': 16, 'weight':'bold'}
    ax.set_xlabel('Weight', fontdict=font)
    ax.set_ylabel('Density', fontdict=font)
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f'weight_density.png',
                dpi=1200, transparent=True)
    plt.show()


def plot_density(features, col, label):
    df = features[[col]]
    ax = plt.gca()
    df.plot(ax=ax, kind='kde', legend=None, fontsize=12,
                 xlim=(df[col].min(), df[col].max()))
    font = {'size': 16, 'weight':'bold'}
    ax.set_xlabel(label, fontdict=font)
    ax.set_ylabel('Density', fontdict=font)
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f'{col}_density.png',
                dpi=1200, transparent=True)
    plt.show()


def plot_features(model):
    """
    Plots feature summary table, density, and heatmap.
    TODO: Merge with the features methods
    :param model: The trained model
    :return: None
    """
    if type(model) == LinearSVC:
        weights = model.coef_.flatten()
    elif type(model) == SVC:
        weights = model.coef_.toarray().flatten()

    df = pd.DataFrame({'weight': weights})
    summary = df.describe()

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig = plt.figure(num='Feature Weights', figsize=(20, 6), facecolor='w')

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 3, 1)
    yl = '#FFFFD1'
    rd = '#F9C1AB'
    gn = '#C5E0B3'
    ax1.axis("off")
    ax1.set_title('Weight Statistics', fontsize=24, fontweight='bold')
    tbl = ax1.table(cellText=np.round(summary.values[1:,:], 4), loc='center',
                    colWidths=[0.6],
                    rowLabels=summary.index[1:], #colLabels=summary.columns,
                    rowColours=[yl, 'w', gn, 'w', 'w', 'w', rd],
                    cellColours=[[yl], ['w'], [gn], ['w'], ['w'], ['w'], [rd]])

    # tbl.auto_set_font_size(False)
    tbl.set_fontsize(24)
    tbl.scale(1, 3.8)

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Density', fontsize=24, fontweight='bold')
    df.plot(ax=ax2, kind='kde', fontsize=20, legend=False,
            xlim=(df.weight.min(), df.weight.max()))
    ax2.set_xlabel('Weight', fontsize=20)
    ax2.set_ylabel('Desnity', fontsize=20)
    # ax2.set(xlabel='', ylabel='', fontsize=20)

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title('Heatmap (100x100=10K features)',
                  fontsize=24, fontweight='bold')
    w = weights.reshape(100, 100)
    GnWtRd = cmap(n_mid=1)
    hmap = sns.heatmap(w, ax=ax3, square=True, cmap=GnWtRd, center=0)
    hmap.collections[0].colorbar.ax.tick_params(labelsize=20)
    for _, spine in hmap.spines.items():
        spine.set_visible(True)
    ax3.set(xlabel='', ylabel='', xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


def weight_violin(features):
    weights = features[['weight']]
    ax = plt.gca()
    plt.violinplot(weights.weight, showmeans=False,
                   showmedians=False, showextrema=False)
    font = {'size': 16, 'weight':'bold'}
    ax.set_xlabel('', fontdict=font)
    ax.set_ylabel('Weight', fontdict=font)
    ax.set_ylim(weights.weight.min(), weights.weight.max())
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f'weight_violin.png',
                dpi=1200, transparent=True)
    plt.show()

def feature_frequency(X, features):
    col_idx = X.tocsc().nonzero()[1]
    counter = np.bincount(col_idx)
    print("Counter:", len(counter))

    zeros = np.where(counter==0)[0]
    print("Zeros count: ", zeros.shape[0])
    print("Zeros pct: ", zeros.shape[0]/features.shape[0])

    ones = np.where(counter==1)[0]
    print("Ones count: ", ones.shape[0])
    print("Ones pct: ", ones.shape[0]/features.shape[0])

    mores = np.where(counter>1)[0]
    print("Mores count: ", mores.shape[0])
    print("Mores pct: ", mores.shape[0]/features.shape[0])


def feature_report(features, X, X_malware, X_goodware, Y,
                   save_csv=False, top=1000):
    print('1. Dataframe of all features\n')
    print(f'Columns:\n--------\n'
          f'0 {"idx":<10} => Index\n'
          f'1 {"feature":<10} => Feature name\n'
          f'2 {"weight":<10} => Feature weight in the trained model\n'
          f'3 {"all_freq":<10} => P(x|D) The normalised frequency of the feature in the whole dataset\n'
          f'4 {"mal_freq":<10} => P(x|malware) The normalised frequency of the feature in the malware class\n'
          f'5 {"good_freq":<10} => P(x|goodware) The normalised frequency of the feature in the goodware class\n'
          f'6 {"chi2":<10} => The chi^2 value to measure the independence between features and classes (for feature selection)\n')
    print(features)
    if save_csv:
        features.to_csv(FEAT_DIR + 'features.csv', sep='\t')

    # Rank by chi2 values, for feature selection
    top_features_chi2 = features.sort_values(by='chi2', ascending=False)
    print(f'\n2. Top {top} features, ranked by Chi2 (for feature selection):')
    print(top_features_chi2.head(top))
    if save_csv:
        top_features_chi2.to_csv(FEAT_DIR + 'top_features_chi2.csv', sep='\t')

    nz_features = features.loc[features.weight != 0]
    print(f'\n3. Features with non-zero weights '
          f'({nz_features.shape[0]}/{features.shape[0]} = '
          f'{nz_features.shape[0]/features.shape[0]*100:.2f}%):')
    print(nz_features)
    if save_csv:
        nz_features.to_csv(FEAT_DIR + 'nz_features.csv', sep='\t')

    # Rank by the magnitude of weights (absolute values)
    ranked_idx = np.argsort(abs(nz_features.weight)).values[::-1]
    abs_features = nz_features.iloc[ranked_idx]
    print(f'\n4. Top {top} features, '
          f'Ranked by weight magnitude (absolute values):')
    print(abs_features.head(top))
    if save_csv:
        abs_features.to_csv(FEAT_DIR + 'abs_nz_features.csv', sep='\t')

    # Rank features with positive weights
    pos_features = nz_features.loc[nz_features.weight > 0]
    pos_features = pos_features.sort_values(by='weight', ascending=False)
    print(f'\n5. Top {top} positive features (out of {pos_features.shape[0]}), '
          f'Ranked by weight:')
    print(pos_features.head(top))
    if save_csv:
        pos_features.to_csv(FEAT_DIR + 'pos_nz_features.csv', sep='\t')

    # Rank features with negative weights
    neg_features = nz_features.loc[nz_features.weight < 0]
    neg_features = neg_features.sort_values(by='weight', ascending=True)
    print(f'\n6. Top {top} negative features (out of {neg_features.shape[0]}), '
          f'Ranked by weight:')
    print(neg_features.head(top))
    if save_csv:
        neg_features.to_csv(FEAT_DIR + 'neg_nz_features.csv', sep='\t')