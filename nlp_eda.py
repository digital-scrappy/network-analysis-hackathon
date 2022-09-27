import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import string

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

import plotly_express as px

from sklearn.decomposition import LatentDirichletAllocation as LDA

tokenizer = RegexpTokenizer(r'\b[A-Za-z0-9\-]{1,}\b')
default_tk = tokenizer
gen_stop_words = list(set(stopwords.words("english")))
gen_stop_words += list(string.punctuation)

from pathlib import Path

def plot_term_freq_dist(df :pd.DataFrame, output_dir : Path, y_term:str='word', top_n:int=20):

    # top terms, filter out
    df = df.sort_values('frequency', ascending=False)
    df = df.iloc[:20]

    fig_follower = px.bar(df, x='frequency',y=y_term,
                        # color= '#followers', color_continuous_scale= 'deep',
                labels={
                    "frequency" : "Frequency",
                    # "#followers" : "number of followers"
                    },
                        )
    fig_follower.update_traces(hovertemplate='Frequency: %{x}'+'<br>Term: %{y}')
    fig_follower.update_layout(title_text= f"Most commonly used {y_term}", title_x= 0.5)
    fig_follower.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig_follower.write_html(output_dir/ f"{top_n}_{y_term}_frequency_plot.html")


#cleaning extract URLs, extract handles

def extract_linkable_features(df:pd.DataFrame, text_col:str='tweet-text')->pd.DataFrame:


    df['extracted_twitter_handles'] = df[text_col].apply(lambda x: re.findall('@[a-zA-Z0-9_]{1,16}', x) if isinstance(x,str) else x)

    url_regex_patt = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df['extracted_URLs'] = df[text_col].apply(lambda x: re.findall(url_regex_patt ,x) if isinstance(x,str) else x)

    df['extracted_hashtags'] = df[text_col].apply(lambda x: re.findall('#[a-zA-Z0-9]{1,140}', x) if isinstance(x,str) else x)

    return df

def remove_linkable_features(df:pd.DataFrame, text_col:str='tweet_text')->pd.DataFrame:

    clean_col = f'clean_{text_col}'
    df[clean_col] = df[text_col].apply(lambda x: re.subn('@[a-zA-Z0-9_]{1,16}','', x)[0] if isinstance(x,str) else x)

    url_regex_patt = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df[clean_col] = df[clean_col].apply(lambda x: re.subn(url_regex_patt ,'',x)[0] if isinstance(x,str) else x)

    df[clean_col] = df[clean_col].apply(lambda x: re.subn('#[a-zA-Z0-9]{1,140}','', x)[0] if isinstance(x,str) else x)

    return df

def extract_and_remove_linkable_features(df:pd.DataFrame, text_col:str='tweet_text'):

    return remove_linkable_features(extract_linkable_features(df, text_col))


def main(input_file:str, output_dir:str):

    return

if __name__=='__main__':

    start_user = sys.argv[0]
    depth = int(sys.argv[1])

    main()