"""Utilitary functions used for text processing in ABES project"""

# Import des librairies
import os
import re

import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

# download nltk packages
nltk.download("words")
nltk.download("stopwords")
nltk.download("omw-1.4")
nlp = spacy.load("fr_core_news_md")

DPI = 300
RAND_STATE = 42


# Set paths
path = "."
os.chdir(path)
data_path = path + "/data"
output_path = path + "/outputs"
fig_path = path + "/figs"


# Set Parameters
RANDOM_STATE = 42


# Import dataset
def get_dataset(filename):
    dataset = pd.read_pickle(
        os.path.join(data_path, filename),
        converters={"DESCR": eval, "rameau_concepts": eval},
    )
    return dataset


# Save Dataset to csv
def save_dataset_to_pickle(df, filename):
    path_destination = os.path.join(data_path, filename)
    df.to_pickle(path_destination, index=False)


# Import processed dataset
def get_dataset_init(filename):
    dataset_init = pd.read_pickle(
        os.path.join(data_path, filename), converters={"rameau_concepts": eval}
    )
    return dataset_init


def flatten(liste):
    flat_list = [item for sublist in liste for item in sublist]
    return flat_list


#                           TEXT PREPROCESS                         #
# --------------------------------------------------------------------
def flatten(liste):
    return [item for sublist in liste for item in sublist]


class PreprocessData:
    def __init__(
        self,
        df,
        input_col,
        output_col,
        lang="french",
        add_words=[],
        encod=None,
        numeric=True,
        stopw=True,
        stem=False,
        lem=True,
    ):
        self.df = df
        self.lang = lang
        self.add_words = add_words
        self.encod = encod
        self.numeric = numeric
        self.stopw = stopw
        self.stem = stem
        self.lem = lem

        if lang == "french":
            self.stop_w = set(stopwords.words(lang)).union(fr_stop).union(add_words)
            ref = "fr_core_news_md"
            self.stemmer = FrenchStemmer()
        elif lang == "english":
            self.stop_w = set(stopwords.words(lang)).union(en_stop).union(add_words)
            ref = "en_core_web_sm"
            self.stemmer = SnowballStemmer(language=lang)
        else:
            ValueError(
                f"Unknown language, must be 'french' or 'english', you provided {lang}"
            )

        self.nlp = spacy.load(ref, disable=["parser", "ner"])

        df_copy = df.copy()
        df_copy[output_col] = df_copy[input_col].apply(
            lambda x: self.preprocess_text(x)
        )
        self.df = df_copy

    def stop_word_filter_fct(self, tokens):
        """
        Description: remove classical french (and optionnally more)
        stopword from a list of words

        Arguments:
            - list_words (list): list of words
            - add_words (list) : list of additionnal words to remove

        Returns :
            - text without stopwords
        """

        stop_w = self.stop_w
        filtered_tokens = [w for w in tokens if w not in stop_w]

        return filtered_tokens

    def funcEnc(self, tokens):
        new_words = []
        for word in tokens:
            new_word = (
                unicodedata.normalize("NFKD", word)
                .encode("ASCII", "ignore")
                .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        return new_words

    def lemma_fct(self, text):
        """
        Description: lemmatize a text

        Arguments:
            - list_words (list): list of words
            - lang (str): language used in the corpus (default: english)

        Returns :
            - Lemmatized list of words
        """
        # if lang == "english":
        #     lemma = WordNetLemmatizer()
        #     lem_w = [
        #         lemma.lemmatize(
        #             lemma.lemmatize(
        #                 lemma.lemmatize(lemma.lemmatize(w, pos="a"), pos="v"), pos="n"
        #             ),
        #             pos="r",
        #         )
        #         for w in list_words
        #     ]

        nlp = self.nlp
        doc = nlp(text)
        lem_w = []
        for token in doc:
            lem_w.append(token.lemma_)

        return lem_w

    def stem_fct(self, tokens):
        """
        Description: Stem a list of words

        Arguments:
            - list_words (list): list of words

        Returns :
            - Stemmed list of words
        """
        stemmer = self.stemmer
        stem_w = [stemmer.stem(w) for w in tokens]

        return stem_w

    def preprocess_text(self, text):
        """
        Description: preprocess a text with different preprocessings.

        Arguments:
            - text (str): text, with punctuation
            - add_words (str): words to remove, in addition to classical english stopwords
            - ascii (bool): whether to transform text into ascii standard (default: False)
            - numeric (bool): whether to remove numerical or not (default: True)
            - stopw (bool): whether to remove classical english stopwords (default: True)
            - stem (bool): whether to stem words or not (default: False)
            - lem (bool): whether to lemmatize words or not (default: True)
            - lang (str): language used in the corpus (default: eng for english). Can be 'eng' or 'fr'.

        Returns :
            - Preprocessed list of tokens
        """

        # Lowerize all words
        text_lower = str.lower(text)

        # remove particular characters and punctuation
        text_lower = re.sub(r"_", " ", text_lower)  # r"_| x |\d+x"

        if self.numeric:
            # remove all numeric characters
            text_lower = re.sub(r"[^\D]", " ", text_lower)

        # Lemmatization
        if self.lem:
            tokens = self.lemma_fct(text_lower)
        else:
            # tokenize
            tokens = word_tokenize(text_lower, language=self.lang)

        if self.encod:
            tokens = self.funcEnc(tokens)

        if self.stopw:
            # remove stopwords
            tokens = self.stop_word_filter_fct(tokens)

        # Stemming
        if self.stem:
            # lemmatization
            tokens = self.stem_fct(tokens)

        # Remove punctuation
        punctuations = "?:!.,;"
        for word in tokens:
            if word in punctuations:
                tokens.remove(word)

        # Finalize text
        self.tokens = tokens
        transf_desc_text = " ".join(tokens)

        return transf_desc_text
