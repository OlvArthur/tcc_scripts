# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
from nltk.corpus import stopwords
import unicodedata
import numpy as np
import re
import warnings
import pandas as pd
import nest_asyncio
import twint
import sys
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


sys.path.insert(1, "/home/arthur/TCC/codigo/twint")
sys.path.insert(1, "twint/")
# print(sys.path)


nest_asyncio.apply()


pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# nltk.download('stopwords')


# %%
c = twint.Config()


# %%
c.Format = "Username: {username} |  Tweet: {tweet}"
c.Limit = 500
c.Pandas = True
c.Username = "mayarinnha"
c.Lang = "pt"
c.Store_csv = True
c.Output = "../../datasets/twitter-questionnaire/"+c.Username+".csv"


# %%
try:
    test = pd.read_csv('/home/arthur/TCC/datasets/twitter-questionnaire/' +
                       c.Username+'.csv', sep=',').filter(items=['username', 'tweet', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN'])
except FileNotFoundError:
    twint.run.Search(c)
    test = pd.read_csv('/home/arthur/TCC/datasets/twitter-questionnaire/' +
                       c.Username+'.csv', sep=',').filter(items=['username', 'tweet', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN'])


# %%
test_copy = test.copy(deep=True)


# %%
# Pattern removal function creation
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# %%
test_copy['tidy_tweets'] = np.vectorize(
    remove_pattern)(test_copy['tweet'], '@[\w]*')
test_copy.tidy_tweets.head(10)


# %%
def _remove_url(data):
    ls = []
    words = ''
    regexp1 = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    regexp2 = re.compile(
        'www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    for line in data:
        urls = regexp1.findall(line)

        for u in urls:
            line = line.replace(u, ' ')

        urls = regexp2.findall(line)

        for u in urls:
            line = line.replace(u, ' ')

        ls.append(line)
    return ls


# %%
test_copy.tidy_tweets = _remove_url(test_copy.tidy_tweets)
test_copy.head(40)


# %%
def remove_punctuation(input_txt):
    input_txt = input_txt.replace('.', ' ').replace('?', ' ').replace('!', ' ').replace(':', ' ').replace(
        ',', ' ').replace(';', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('#', ' ')
    return input_txt


# %%
test_copy['tidy_tweets'] = test_copy['tidy_tweets'].apply(
    lambda x: remove_punctuation(x))
test_copy.head(10)


# %%
test_copy.tidy_tweets = test_copy.tidy_tweets.str.lower()
test_copy.head()


# %%
# test_copy.tidy_tweets = test_copy.tidy_tweets.apply(lambda x: unicodedata.normalize('NFD',x).encode('ascii','ignore').decode('utf8') )
# test_copy.head(10)


# %%
tokenized_tweet = test_copy.tidy_tweets.apply(lambda x: x.split())
tokenized_tweet.head(10)


# %%
def _apply_standardization(tokens, std_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []
        for word in tk_line:
            if word in std_list:
                word = std_list[word]

            new_tokens.append(word)

        ls.append(new_tokens)

    return ls


# %%
std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês', 'tb': 'também', 'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente',
            'q': 'que', 'n': 'não', 'cmg': 'comigo', 'p': 'para', 'ta': 'está', 'to': 'estou', 'vdd': 'verdade', 'pra': 'para', 'pro': 'para'}


# %%
test_copy['tidy_tweets'] = _apply_standardization(tokenized_tweet, std_list)
test_copy.tidy_tweets.head()


# %%
pt_stopwords = stopwords.words('portuguese')
stop_words = []
noisy_words = ['.', '?', '!', ':', ',', ';', '(', ')', '-']

stop_words.extend(pt_stopwords)
stop_words.extend(noisy_words)

stop_words = list(set(stop_words))


# %%
test_copy.tidy_tweets = test_copy.tidy_tweets.apply(
    lambda x: [word for word in x if word not in stop_words])
test_copy.tidy_tweets.head(40)


# %%
test_copy.head(10)
