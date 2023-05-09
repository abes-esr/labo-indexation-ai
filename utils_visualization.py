"""Utilitary functions for vizualisation used in ABES project"""

# Import des librairies
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud

DPI = 300

# Set paths
path = "."
os.chdir(path)
data_path = path + "/data"
output_path = path + "/outputs"
fig_path = path + "/figs"


#                            EXPLORATION                             #
# --------------------------------------------------------------------
def plot_wordcloud(
    keywords,
    backgound_color="white",
    figsize=(15, 8),
    width=1000,
    height=500,
    save_file=None,
):
    plt.figure(figsize=figsize)
    wordcloud = WordCloud(
        width=width, height=height, background_color=backgound_color
    ).generate_from_frequencies(Counter(keywords))
    plt.imshow(wordcloud)
    if save_file:
        plt.savefig(os.path.join(fig_path, save_file), dpi=DPI, bbox_inches="tight")
    else:
        plt.show()
        plt.close()


def plot_barplot_of_tags(
    tags_list,
    nb_of_tags,
    xlabel="Nombre d'occurences",
    ylabel="",
    figsave=None,
    figsize=(10, 30),
    palette="viridis",
    orient="h",
):
    """
    Description: plot barplot of tags count (descending order) from a list of tags

    Arguments:
        - tags_list (lsit): list of tags
        - nb_of_tags (int) : number of tags to plot in barplot (default=50)
        - xlabel, ylabel (str): labels of the barplot
        - figsize (list) : figure size (default : (10, 30))
        - palette (str): color palette to use (default: viridis)
        - orient (str): orientation of the plot (default:"h")

    Returns :
        - Barplot of nb_of_tags most important tags

    """
    tag_count = Counter(tags_list)
    tag_count_sort = dict(tag_count.most_common(nb_of_tags))

    plt.figure(figsize=figsize)
    sns.barplot(
        x=list(tag_count_sort.values()),
        y=list(tag_count_sort.keys()),
        orient=orient,
        palette=palette,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if figsave:
        plt.savefig(os.path.join(fig_path, figsave), bbox_inches="tight")
    plt.show()


def metrics_radar_plot(
    summary,
    metrics=[
        "Hamming Loss",
        "Accuracy",
        "F1_Score - Sample",
        "Precision - Sample",
        "Recall - Sample",
        "Jaccard - Sample",
    ],
    remove_identity=True,
    title=None,
    savefig=None,
    width=800,
    height=550,
):
    """
    Show values of clusters' centroids on each variable

    Parameters:
    -----------
        - summary (pd.DataFrame): summary with results of optimized models
        - metrics (list of strings): list of metrics to plot
        - remove_identity (bool): whether to remove identical comparisons or not (default: True)
        - title (str): title to write on plot
        - savefig (str): name of the figure if wanted to save (default: no saving)
        - width (int): width of the plot (default:650)
        - height (int) : height of the plot (default:500)

    Returns :
    ---------
        - Radar plot of standardized metrics
    """

    fig = go.Figure()
    subset = summary[metrics]
    if remove_identity:
        comp = [s.split("-") for s in subset.index]
        is_not_identity = [compar[0] != compar[1] for compar in comp]
        subset = subset[is_not_identity]

    if "Hamming Loss" in metrics:
        subset.loc[:, "1-Hamming_Loss"] = 1 - subset.loc[:, "Hamming Loss"]
        subset.drop(columns="Hamming Loss", inplace=True)
    scale_subset = MinMaxScaler().fit_transform(subset)
    for i, idx in enumerate(subset.index):
        fig.add_trace(
            go.Scatterpolar(
                r=scale_subset[i], theta=subset.columns, fill="toself", name=idx
            )
        )
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            autosize=True,
            width=width,
            height=height,
        )
    if savefig:
        fig.write_html(os.path.join(fig_path, savefig))

    fig.show()
