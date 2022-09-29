from cgitb import text
from lib2to3.pytree import generate_matches
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
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
import plotly_express as px

from sklearn.decomposition import LatentDirichletAllocation as LDA

tokenizer = RegexpTokenizer(r'\b[A-Za-z0-9\-]{1,}\b')
default_tk = tokenizer
# gen_stop_words = list(set(stopwords.words("english")))
# gen_stop_words += list(set(stopwords.words('french')))
# gen_stop_words += list(set(stopwords.words('german')))
# gen_stop_words += list(set(stopwords.words('spanish')))
# gen_stop_words += list(set(stopwords.words('russian')))

# gen_stop_words += list(string.punctuation)

from pathlib import Path

def generate_stop_words_by_language(lst_languages:list)->list:
    """Returns a list with stopwords from different languages, based on 
    nltk.corpus.stopwords. Also adds punctuations to the list of stopwords. 
    Result to be used in data cleaning/omitting terms

    Args:
        lst_languages (list): list of languages to use. If an incorrect language 
        is specified, then it will be omitted but an error will NOT be raised

    Returns:
        list: list of stopwords 
    """    

    gen_stop_words = []
    for lang in lst_languages:
        try:
            gen_stop_words += list(set(stopwords.words(lang)))

        except OSError: 
            print(f'Could not find stop words for {lang}. Omitting from list of stopwords')
            pass

    gen_stop_words += list(string.punctuation)

    return gen_stop_words

lst_languages = ['english', 'french', 'spanish', 'german', 'spanish', 'russian']
gen_stop_words = generate_stop_words_by_language(lst_languages)


def apply_tfidf_and_return_table_of_results(tfidf:TfidfVectorizer, df:pd.DataFrame, text_col:str)->pd.DataFrame:
    """Fn takes in dataframe, with specified text column, an instantiated sklearn tfidf-vectorizer 
    and outputs a sorted table of all the the terms with their respective tf-idf scores

    Args:
        tfidf (TfidfVectorizer): sklearn tfidf vectorizer instance
        df (pd.DataFrame): dataframe
        text_col (str): text column we wish to tokenizer and analyse

    Returns:
        pd.DataFrame: single-column table containing the terms as an index 
    """    
    tfidf_df = pd.DataFrame(tfidf.fit_transform(df['clean_tweet_text']).toarray(), index = df.index, columns = tfidf.get_feature_names_out())

    # full_df = df.join(new_df)
    tfidf_sum = pd.DataFrame(tfidf_df.sum(), columns = ['tf_idf_score']).sort_values('tf_idf_score', ascending=False).reset_index().rename({'index':'terms'})
    
    return tfidf_sum

def get_tfidf_scores(df:pd.DataFrame, text_col:str, ngram_range:tuple=(1,2), 
                    tokenizer=tokenizer.tokenize, stopwords:list=gen_stop_words, 
                    min_doc_frequency:float=0.1, max_doc_frequency:float = 1.0,
                    smooth_idf:bool=False,
                    )->pd.DataFrame:


    tfidf = TfidfVectorizer(tokenizer=tokenizer, 
                            ngram_range=ngram_range, 
                            min_df=min_doc_frequency, 
                            max_df=max_doc_frequency,
                            smooth_idf=smooth_idf,
                            stop_words=stopwords)

    return apply_tfidf_and_return_table_of_results(tfidf, df, text_col)


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
    fig_follower.update_layout(title_text= f"{top_n} most commonly used {y_term}", title_x= 0.5)
    fig_follower.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig_follower.write_html(output_dir/ f"{top_n}_{y_term}_frequency_plot.html")


def tokenize_text(df:pd.DataFrame, text_col:str='clean_tweet_text', stopwords:list=gen_stop_words)->pd.DataFrame:
    df[f'{text_col}_tokens'] = df[text_col].apply(lambda x : [tok for tok in tokenizer.tokenize(x) if tok not in stopwords])

def print_top_trig_collocs(pd_series:pd.Series, tokenizer, frac_corpus = 0.1, stopwords = gen_stop_words):
    corpus = [tokenizer.tokenize(x) for x in pd_series.to_list()]
    finder = TrigramCollocationFinder.from_documents(corpus)
    finder.apply_freq_filter(round(frac_corpus*len(pd_series)))
    main_trigrams = finder.nbest(trigram_measures.likelihood_ratio, 100000)
    # for trigram in main_trigrams:
    #     if word in trigram:
    #         print(trigram)
        
    return main_trigrams

def print_top_bigr_collocs(pd_series:pd.Series, tokenizer, frac_corpus = 0.1, stopwords = gen_stop_words):
    corpus = [tokenizer.tokenize(x) for x in pd_series.to_list()]
    finder = BigramCollocationFinder.from_documents(corpus)
    finder.apply_freq_filter(round(frac_corpus*len(pd_series)))
    main_bigrams = finder.nbest(bigram_measures.likelihood_ratio, 100000)
    # for trigram in main_trigrams:
    #     if word in trigram:
    #         print(trigram)
        
    return main_bigrams

def get_term_freq_df(tok_series:pd.Series)->pd.DataFrame:
    """Takes in a pandas series with tokenized cells of text; adds them together
    then returns the frequency distribution dataframe for that corpus. 

    Args:
        tok_series (pd.Series): data series containing tokenized text

    Returns:
        pd.DataFrame: dataframe with corpus frequency distribution values
    """    

    corpus_lst = tok_series.to_list()
    joined_corpus = []
    for doc in corpus_lst:
        for token in doc:
            joined_corpus.append(token)

    fdist = FreqDist(joined_corpus)

    term_freqs = pd.DataFrame(fdist, index=[0]).T.reset_index()
    term_freqs.columns = ['term', 'frequency']
    term_freqs.sort_values('frequency', ascending=False, inplace=True)
        
    return term_freqs

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