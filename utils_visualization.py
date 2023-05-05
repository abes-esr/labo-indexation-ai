"""Utilitary functions for vizualisation used in ABES project"""

# Import des librairies
import os
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

DPI = 300
RAND_STATE = 42

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
    save_file=None
    ):

    plt.figure(figsize=figsize)
    wordcloud = WordCloud(
        width=width, height=height,
        background_color=backgound_color).generate_from_frequencies(Counter(keywords))
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
    orient="h"
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
