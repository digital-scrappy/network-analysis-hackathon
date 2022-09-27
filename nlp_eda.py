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



def main(input_file:str, output_dir:str):

    return

if __name__=='__main__':

    start_user = sys.argv[0]
    depth = int(sys.argv[1])

    main()