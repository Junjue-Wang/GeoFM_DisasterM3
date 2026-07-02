import matplotlib.pyplot as plt
import itertools
import numpy as np

__all__ = [
    'plot_confusion_matrix',
    'plot_confusion_matrix_by_seaborn'
]


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          figsize=(8, 5),
                          title='Confusion Matrix',
                          x_label='Predicted label',
                          y_label='True label',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    return fig


def plot_confusion_matrix_by_seaborn(cm,
                                     class_names,
                                     normalize=True,
                                     title='Confusion Matrix',
                                     x_label='Predicted label',
                                     y_label='True label',
                                     cmap=plt.cm.Blues,
                                     rotation=30,
                                     **kwargs
                                     ):
    import seaborn as sns
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        format = '.3f'
    else:
        format = 'd'
    ax = sns.heatmap(cm, annot=True, fmt=format, annot_kws={"size": 12}, cmap=cmap,
                     xticklabels=class_names,
                     yticklabels=class_names,
                     **kwargs)
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    return ax
