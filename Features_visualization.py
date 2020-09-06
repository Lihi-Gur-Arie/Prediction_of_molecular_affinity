# by Lihi Gur-Arie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def tsne_plot(data, title, perplexity, random_state, n_iter=1000, save=False):
    """
    Dimensionality reduction via t-SNE, 2 dimensional plot.

    :param data: pd.DataFrame
        Data, with the label at the last column.
    :param title: str
        The plot's title
    :param perplexity: int
        t-SNE hyper-parameter
    :param random_state: int
        The seed for the ramdom state
    :param n_iter: int
        Number of Iterations
    :param save: bool
        If True, the plot will be saved
    """
    X = data.iloc[:, :-1]

    # t-SNE algo:
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, random_state=random_state, n_iter=n_iter, n_jobs=-1)
    tsne_results = pd.DataFrame(tsne.fit_transform(X), columns=["tsne1", "tsne2"])

    # plot t-SNE:
    cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True, reverse=True)  # Set the colors
    f, ax = plt.subplots()
    plt.xlabel('t-SNE diamention 1', fontsize=14)
    plt.ylabel('t-SNE diamention 2', fontsize=14)
    ax.axes.xaxis.set_ticks([])  # Remove X ticks
    ax.axes.yaxis.set_ticks([])  # Remove Y ticks
    points = ax.scatter(tsne_results.iloc[:, 0], tsne_results.iloc[:, 1], c=data.iloc[:, -1], s=70, cmap=cmap)
    f.colorbar(points)
    plt.title(title, fontsize=20)

    # Save plot:
    if save is True:
        plt.savefig(f"TSNE_plot_{title}_{perplexity}_{random_state}_{n_iter}.png", bbox_inches='tight', dpi=600)
    plt.show()


def d2_scatter_plot(xs, ys, color, title, save=False):
    """
    Two dimensional scatter plot

    :param xs: pd.Series
        The date for X axis

    :param ys: pd.Series
        The date for Y axis

    :param color: pd.Series
        The date to be colored by

    :param title: str
        The plot's title

    :param save: bool
        Save the plot
    """

    plt.scatter(x=xs, y=ys, c=color, cmap='cool', alpha=1)  # Try cmap='GnBu_r'
    plt.title(title)
    plt.xlabel(xs.name)
    plt.ylabel(ys.name)
    plt.colorbar()
    if save is True:
        plt.savefig(f"2D_scatter_plot_{title}.png", bbox_inches='tight', dpi=600)
    plt.show()


def d3_scatter_plot(xs, ys, zs, color, title, save=False):
    """
    Three dimensional scatter plot

    :param xs: pd.Series
        The date for X axis
    :param ys: pd.Series
        The date for Y axis
    :param zs: pd.Series
        The date for Y axis
    :param color: pd.Series
        The date to be colored by
    :param title: str
        The plot's title
    :param save: bool
        If True, the plot will be saved
    """

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    p = ax.scatter3D(xs=xs, ys=ys, zs=zs, c=color, cmap='cool', alpha=1)  # Try cmap='GnBu_r'
    fig.colorbar(p, ax=ax)
    plt.title(title)
    ax.set_xlabel(xs.name)
    ax.set_ylabel(ys.name)
    ax.set_zlabel(zs.name)
    # Save plot:
    if save is True:
        plt.savefig(f"3D_scatter_plot_{title}.png", bbox_inches='tight', dpi=600)
    plt.show()


def correlation_map(data, title, save):
    """
    Plot correlation map

    :param data: pd.DataFrame
        The data for the plot
    :param title: str
        The plot's title
    :param save: bool
        If True, the plot will be saved
    """

    # make a plot of specified dimension (in inches)
    fig, ax = plt.subplots(figsize=(25, 20))
    # pass the axis to draw on
    sns.set(font_scale=3)
    matrix = np.triu(data.corr())
    b = sns.heatmap(data.corr(), annot=True, annot_kws={"size": 14}, fmt='.1g', center=0,
                    cmap=sns.diverging_palette(220, 20, as_cmap=True), mask=matrix)
    b.axes.set_title("Correlation Map", fontsize=50)
    b.tick_params(labelsize=17)
    # Save plot:
    if save is True:
        plt.savefig(f"Correlation_map_{title}_.png", bbox_inches='tight', dpi=600)
    plt.show()


def train_vs_test_plot(title, train_data, test_data, plot_type="boxplot", save=False):
    """
    Plot train data vs. test data

    :param title: str
        The plot's title
    :param train_data: pd.DataFrame
        The train data
    :param test_data: pd.DataFrame
        The test data
    :param plot_type: str
        'boxplot' or 'swarmplot'
    :param save: bool
        If True, the plot will be saved

    """
    # add y_class category:
    train_data['train_test'] = 1
    test_data['train_test'] = 0
    all_data = pd.concat([train_data, test_data])
    melted = all_data.melt(id_vars=['train_test'])

    # plot:
    sns.set_style("darkgrid")
    plt.figure(figsize=(25, 10))
    plt.title(title, fontsize=30)
    sns.set(font_scale=1.5, palette="pastel", color_codes=True)

    if plot_type == 'boxplot':
        ax = sns.boxplot(x="variable", y="value", hue="train_test", data=melted)
    elif plot_type == 'swarmplot':
        ax = sns.swarmplot(x="variable", y="value", hue="train_test", data=melted, dodge=True, size=2)

    ax.set_xticklabels(labels=train_data.columns.values, rotation='vertical', fontsize=15)

    # Add legend:
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Test", "Train"])

    plt.xlabel('Features', fontsize=22)
    plt.ylabel('Feature value (standard scaled)', fontsize=22)

    # Save plot:
    if save is True:
        plt.savefig(f"Train_vs_test_plot_{title}_.png", bbox_inches='tight', dpi=600)
    plt.show()
    print('plot is ready')


def divided_violin_plot(title, x_scaled_standard, y_unscaled, save=False):
    """
    Violin plot of all the features, splited by Y threshold of 13.2

    :param title: str
        The plot's title
    :param x_scaled_standard: pd.DataFrame
        The data after standard scaling
    :param y_unscaled: pd.Series
        The label.
    :param save: bool
        If True, the plot will be saved
    """
    # add y_class category:
    x_scaled_standard['Y_class'] = (np.array([1.0 if y > -13.2 else -1.0 for y in y_unscaled.squeeze()])).reshape(-1, 1)
    melted = x_scaled_standard.melt(id_vars=['Y_class'])

    # plot:
    sns.set_style("darkgrid")
    plt.figure(figsize=(25, 7))
    plt.title(title, fontsize=30)

    sns.set(font_scale=1.5, palette="pastel", color_codes=True)
    ax = sns.violinplot(x="variable", y="value", hue="Y_class", data=melted, split=True, bw=.2)
    ax.set_xticklabels(labels=x_scaled_standard.columns.values, rotation='vertical', fontsize=15)

    # Add ledend:
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Score <= -11.25", "Score > -13.2"])
    plt.xlabel('Features', fontsize=22)
    plt.ylabel('Feature value (standard scaled)', fontsize=22)

    # Save plot:
    if save is True:
        plt.savefig(f"Features_violin_plot_{title}_.png", bbox_inches='tight', dpi=600)
    plt.show()
    print('Violin_plot is ready')


def data_vs_mol_plots(title, first_data, first_plot_type="boxplot", second_data=None, second_plot_type='swarmplot',
                      save=False):
    """
    Plot data vs. specific molecules

    :param title: str
        The plot's title
    :param first_data: pd.DataFrame
        The data after standard scaling
    :param first_plot_type: str
        "boxplot" or "swarmplot"
    :param second_data: pd.DataFrame
        The molecules to examine.
    :param second_plot_type: str
        'stripplot' or 'swarmplot'
    :param save: bool
        If True, the plot will be saved
    """

    # New plot:
    sns.set_style("darkgrid")
    plt.figure(figsize=(25, 7))
    plt.title(title, fontsize=30)

    # Choose the plot type of the first data_origin:
    if first_plot_type == "swarmplot":
        ax = sns.swarmplot(data=first_data)  # color=sns.xkcd_rgb["green blue"])
    else:
        ax = sns.boxplot(data=first_data)  # color=sns.xkcd_rgb["green blue"]

    ax.set_xticklabels(labels=first_data.columns.values, rotation='vertical', fontsize=15)

    plt.xlabel('Features', fontsize=22)
    plt.ylabel('Feature value (standard scaled)', fontsize=22)

    # Choose the second plot's type:
    if second_data is not None:
        # Plot the dots:
        second_data['mol_index'] = second_data.index.values
        melted = second_data.melt(id_vars=['mol_index'])
        melted.rename(columns={"variable": "Feature", "value": "Feature_value"})

        # melted = melted[melted.mol_index == second_data.index.values[423]]

        if second_plot_type == 'stripplot':
            stp = sns.stripplot(x="variable", y="value", hue='mol_index', data=melted, jitter=True, palette="Set2",
                                linewidth=1, edgecolor='gray')
        else:
            stp = sns.swarmplot(x="variable", y="value", hue='mol_index', data=melted, palette="Set2", linewidth=1,
                                edgecolor='gray', size=4)

        # Add legend:
        handles, _ = stp.get_legend_handles_labels()
        stp.legend(handles, second_data.index.values)

    if save == True:
        plt.savefig(f"Features_plot_{title}.png", bbox_inches='tight', dpi=600)
    plt.show()
    print('Plot is ready')
