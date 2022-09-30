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
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

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


class LemmaTokenizer(object):
    def __init__(self, tokenizer = default_tk, stopwords = gen_stop_words):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = tokenizer
        self.stopwords = stopwords
    def __call__(self, articles):
        return [self.wnl.lemmatize(token, ) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    
    def tokenize(self, articles):
        return [self.wnl.lemmatize(token) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    

lemmy = LemmaTokenizer()

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
    tfidf_df = pd.DataFrame(tfidf.fit_transform(df[text_col]).toarray(), index = df.index, columns = tfidf.get_feature_names_out())

    # full_df = df.join(new_df)
    tfidf_sum = pd.DataFrame(tfidf_df.sum(), columns = ['tf_idf_score']).sort_values('tf_idf_score', ascending=False).reset_index().rename({'index':'terms'}, axis=1)
    
    return tfidf_sum

def apply_count_vect_and_return_table_of_results(cvt:CountVectorizer, df:pd.DataFrame, text_col:str)->pd.DataFrame:
    """Fn takes in dataframe, with specified text column, an instantiated sklearn count-vectorizer 
    and outputs a sorted table of all the the terms with their respective tf-idf scores

    Args:
        cvt (CountVectorizer): sklearn count vectorizer instance
        df (pd.DataFrame): dataframe
        text_col (str): text column we wish to tokenizer and analyse

    Returns:
        pd.DataFrame: single-column table containing the terms as an index 
    """    
    cvt_df = pd.DataFrame(cvt.fit_transform(df[text_col]).toarray(), index = df.index, columns = cvt.get_feature_names_out())

    
    # cvt_sum = pd.DataFrame(cvt_df.sum(), columns = ['cvt_score']).sort_values('cvt_score', ascending=False).reset_index().rename({'index':'terms'}, axis=1)
    
    return cvt_df

def apply_tfidf_and_return_grouped_table_of_results(tfidf:TfidfVectorizer, df:pd.DataFrame, text_col:str, group_col:str)->pd.DataFrame:
    """Fn takes in dataframe, with specified text column, an instantiated sklearn tfidf-vectorizer 
    and outputs a sorted table of all the the terms with their respective tf-idf scores 
    but aggregated by the level of the group col specified
    Args:
        tfidf (TfidfVectorizer): sklearn tfidf vectorizer instance
        df (pd.DataFrame): dataframe
        text_col (str): text column we wish to tokenizer and analyse
        group_col(str): column(s) specifying at which level we wish to aggregate, typically
        at the user level. 

    Returns:
        pd.DataFrame: single-column table containing the terms as an index 
    """    
    tfidf_df = pd.DataFrame(tfidf.fit_transform(df[text_col]).toarray(), index = df.index, columns = tfidf.get_feature_names_out())

    if isinstance(group_col, str):
        group_col = [group_col]

    full_df = df[group_col].join(tfidf_df)
    agg_df = get_tfidf_grouped_scores(df, text_col, group_col, 
                                     )

    agg_df_melt = agg_df.groupby(group_col).sum().reset_index().melt(id_vars=[group_col], 
                        var_name='term', value_name='tfidf_score').sort_values([ group_col, 'tfidf_score'], ascending=False)

    # tfidf_sum = pd.DataFrame(tfidf_df.sum(), columns = ['tf_idf_score']).sort_values('tf_idf_score', ascending=False).reset_index().rename({'index':'terms'}, axis=1)
    
    return agg_df_melt

def get_tfidf_grouped_scores(df:pd.DataFrame, text_col:str,group_col:str, ngram_range:tuple=(1,2), 
                    tokenizer=tokenizer.tokenize, stopwords:list=gen_stop_words, 
                    min_doc_frequency:float=0.01, max_doc_frequency:float = 1.0,
                    smooth_idf:bool=False,
                    )->pd.DataFrame:


    tfidf = TfidfVectorizer(tokenizer=tokenizer, 
                            ngram_range=ngram_range, 
                            min_df=min_doc_frequency, 
                            max_df=max_doc_frequency,
                            smooth_idf=smooth_idf,
                            stop_words=stopwords)

    return apply_tfidf_and_return_grouped_table_of_results(tfidf, df, text_col, group_col)

def get_tfidf_scores(df:pd.DataFrame, text_col:str, ngram_range:tuple=(1,2), 
                    tokenizer=tokenizer.tokenize, stopwords:list=gen_stop_words, 
                    min_doc_frequency:float=0.01, max_doc_frequency:float = 1.0,
                    smooth_idf:bool=False,
                    )->pd.DataFrame:


    tfidf = TfidfVectorizer(tokenizer=tokenizer, 
                            ngram_range=ngram_range, 
                            min_df=min_doc_frequency, 
                            max_df=max_doc_frequency,
                            smooth_idf=smooth_idf,
                            stop_words=stopwords)

    return apply_tfidf_and_return_table_of_results(tfidf, df, text_col)

def get_count_vectorized_df(df:pd.DataFrame, text_col:str, ngram_range:tuple=(1,2), 
                    tokenizer=lemmy, stopwords:list=gen_stop_words, 
                    min_doc_frequency:float=0.01, max_doc_frequency:float = 1.0,
                    )->pd.DataFrame:


    count_vect = CountVectorizer()
    count_vect = CountVectorizer(tokenizer=tokenizer, 
                            ngram_range=ngram_range, 
                            min_df=min_doc_frequency, 
                            max_df=max_doc_frequency,
                            stop_words=stopwords)

    return apply_count_vect_and_return_table_of_results(count_vect, df, text_col)


def calculate_and_plot_tfidf(input_dir:Path, output_dir:Path, top_n:int, text_col_raw :str, tokenizer=tokenizer.tokenize, 
                            stopwords:list=gen_stop_words, 
                            ngram_range:tuple=(1,2),
                        min_doc_frequency:float=0.01, max_doc_frequency:float = 1.0,
                        smooth_idf:bool=False,):
    """Pulls up the csv file containing the tweet texts, cleans, tokenizes and vectorizes
    the text data and outputs (and returns) a plotly_express graph of the top_n terms by 
    tf-idf score.

    Args:
        input_dir (Path): directory containing tweet_text.csv
        output_dir (Path): directory where we want to output the results (normally the same as input_dir)
        top_n (int): how many terms do we want displayed
        text_col_raw (str): name of raw text column
        tokenizer (optional): Tokenizer for parsing and tokenizing text. Defaults to tokenizer.tokenize.
        stopwords (list, optional): Stopwords to be discarded. Please modify the generate_stop_words() function
        to add more stopwords from different languages. Punctuation is also includedd. Defaults to gen_stop_words.
        ngram_range (tuple, optional): N grams we want to look at. If we want just single tokens, then
        specify (1,1). Just bigrams : (2,2). Uni-grams, bigrams and trigrams : (1,3). Defaults to (1,2).
        min_doc_frequency (float, optional): The minimum fraction of tweets we want a term to appear in for it
        to be included in the final table. Defaults to 0.01 (i.e. 1% of tweets).
        max_doc_frequency (float, optional): The maximum fraction of tweets we want a term to appear in for it 
        to be included in the final table - this is useful in case we want to cut-off terms that appear almost
        everywhere (but remember that the tf-idf score in itself already does some of the work in reducing/eliminating
        those terms). Defaults to 1.0 (i.e. 100%).
        smooth_idf (bool, optional): Whether or not to add 1 to the denominator. This is only useful to have as 
        True when using the tf-idf vectorizer for new, unseen data (e.g. in the case of building a model).
        The canonical form of the tf-idf formula that most researchers expected, however, does *NOT* have a 1
        added to the denominator, so it's recommended for descriptive stats that we keep this set to default.
         Defaults to False.

    Returns:
        plot: plotly_express bar plot
    """    
    df = etl_tweet_text(input_dir)
    text_col_clean = f'clean_{text_col_raw}'
    tfidf_ = get_tfidf_scores(df, text_col_clean, ngram_range, tokenizer, stopwords, min_doc_frequency, max_doc_frequency, smooth_idf)
    return plot_tfidf_dist(tfidf_, output_dir, top_n)

def etl_tweet_text(input_dir:Path, text_col_raw:str='tweet_text')->pd.DataFrame:
    """Get csv file from input_dir tweet_text.csv, extract and clean features such as 
    URLs, mentions and hashtags into separate columns. Returns copy of data with
    raw text col and new clean version (same name as raw column with 'clean_' prefix)

    Args:
        input_dir (Path): directory containing tweet_text.csv
        output_dir (Path): directory where we want to output the results (normally the same as input_dir)
        text_col_raw (str): name of raw text column
        
    Returns:
        pd.DataFrame: clean dataframe
    """    
    fpath = input_dir/ Path('tweet_text.csv')
    df = pd.read_csv(str(fpath))
    df = extract_and_remove_linkable_features(df, text_col_raw)
    return df

# def 


def plot_tfidf_dist(data :pd.DataFrame, output_dir : Path,  top_n:int=20):

    # top terms, filter out
    df = data.copy()
    df = df.sort_values('tf_idf_score', ascending=False)
    df = df.iloc[:top_n]

    top_n = len(df)

    fig_follower = px.bar(df, x='tf_idf_score',y='terms',
                        # color= '#followers', color_continuous_scale= 'deep',
                labels={
                    "frequency" : "Frequency",
                    # "#followers" : "number of followers"
                    },
                        )
    fig_follower.update_traces(hovertemplate='tf-idf score: %{x}'+'<br>Term: %{y}')
    fig_follower.update_layout(title_text= f"Top {top_n} terms by TF-IDF* scores", title_x= 0.5,
                                # subtitle_text = f"*Term Frequency Inverse Document Frequency - a measure of the importance\nor relevance of the term"
                                )
    fig_follower.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig_follower.write_html(output_dir/ f"{top_n}_TF_IDF_frequency_plot.html")
    return fig_follower


def plot_term_freq_dist(data :pd.DataFrame, output_dir : Path, y_term:str='word', top_n:int=20):

    df = data.copy()
    # top terms, filter out
    df = df.sort_values('frequency', ascending=False)
    df = df.iloc[:top_n]

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

def get_lda_topic_data(input_dir:Path, text_col_raw:str='tweet_text', 
                        lemmatizer=lemmy,  **kwargs):

    
    #etl of data

    df = etl_tweet_text(input_dir)
    text_col_clean = f'clean_{text_col_raw}'

    # lemmatize the text col
    # lemm_col = f'lemmatized_{text_col_raw}'
    # df[lemm_col] = df[text_col_clean].apply(lambda x : lemmatize_text_data(x))

    # count vectorize data

    cvt_df = get_count_vectorized_df(df, text_col_clean, tokenizer = lemmatizer)

    #get lda object
    lda_model_ = inst_lda_object(**kwargs)
    # return cvt_df
    lda_df, lda_model_ = dt_to_lda(cvt_df, lda_model_)

    return lda_df, lda_model_

def lemmatize_text_data(x:str, lemmatizer=lemmy)->pd.DataFrame:
    return ' '.join(lemmy(x))

def inst_lda_object(**kwargs):
    try:
        alpha_ = kwargs['alpha']
        eta_ = kwargs['eta']
        n_topics_ = kwargs['n_topics']

    except KeyError:
        n_topics_ = 10
        alpha_ = 1/n_topics_
        eta_ = 1/n_topics_

    return LDA(n_components=n_topics_,
             doc_topic_prior=alpha_, 
             topic_word_prior=eta_)

def dt_to_lda(data, lda_obj):
    """Takes document term matrix and returns (dataframe with LDirA topic data, 
    LDirA sklearn object). Specify number of non-DocTerm columns, fn will assume 
    all the Doc-Term columns are to the left of that. 
    Params:
    data - (Pandas DataFrame obj) dataframe containing text and other data
    your original dataframe, pre-vectorisation.
    lda_obj - (obj) pre-instantiated sklearn LatentDirichletAllocation object
    
    Returns: 
    new_df , vect_train_df (tuple) - dataframe with/out non-text columns on the left and document term matrix on the right"""
    
    
    lda_df = pd.DataFrame(lda_obj.fit_transform(data), index=data.index, columns=list(range(1,(lda_obj.n_components+1))))
    lda_df = lda_df.add_prefix('topic_')
    
    return lda_df, lda_obj

def preprocess_data(string):
    """Function that takes in any single continous string;
    Returns 1 continuous string
    A precautionary measure to try to remove any emails or websites that BS4 missed"""
    new_str = re.sub(r"\S+@\S+", '', string)
    new_str = re.sub(r"\S+.co\S+", '', new_str)
    new_str = re.sub(r"\S+.ed\S+", '', new_str)
    new_str_tok = tokenizer.tokenize(new_str)
    new_str_lemm = [lemmy.lemmatize(token) for token in new_str_tok]
    new_str_cont = ''
    for tok in new_str_lemm:
        new_str_cont += tok + ' '
    return new_str_cont

def print_top_trig_collocs(pd_series:pd.Series, tokenizer, frac_corpus = 0.1, stopwords = gen_stop_words):
    corpus = [tokenizer.tokenize(x) for x in pd_series.to_list()]
    finder = TrigramCollocationFinder.from_documents(corpus)
    finder.apply_freq_filter(round(frac_corpus*len(pd_series)))
    main_trigrams = finder.nbest(trigram_measures.likelihood_ratio, 100000)
    # for trigram in main_trigrams:
    #     if word in trigram:
    #         print(trigram)
        
    return main_trigrams

def print_top_bigr_collocs(pd_series:pd.Series, tokenizer, frac_corpus = 0.01, stopwords = gen_stop_words):
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

def extract_linkable_features(df:pd.DataFrame, text_col:str='tweet_text')->pd.DataFrame:


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