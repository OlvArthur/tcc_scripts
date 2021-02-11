# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from gensim.models.keyedvectors import KeyedVectors
from preprocessing import train_copy
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
import arff

import sys
sys.path.insert(1, '/home/arthur/TCC/codigo')


# %%
train_copy.head()


# %%
# Converting glove format into word2vec format
glove2word2vec('../GloveDatasets/English/glove.twitter.27B.200d.txt',
               '../GloveDatasets/English/glove.twitter.27B.200d.txt.word2vec')


# %%
# Loading word embedding model
model = KeyedVectors.load_word2vec_format(
    '../GloveDatasets/English/glove.twitter.27B.200d.txt.word2vec', binary=False)


# %%
tokenized_status = train_copy.tidy_status
tokenized_status[20:30]


# %%
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# %%
wordvec_arrays = np.zeros((len(tokenized_status), 200))
for i in range(len(tokenized_status)):
    wordvec_arrays[i, :] = word_vector(tokenized_status[i], 200)
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape


# %%
# EXT_wordvec_df = wordvec_df.join(train_copy.cEXT)
# NEU_wordvec_df = wordvec_df.join(train_copy.cNEU)
# AGR_wordvec_df = wordvec_df.join(train_copy.cAGR)
# CON_wordvec_df = wordvec_df.join(train_copy.cCON)
# OPN_wordvec_df = wordvec_df.join(train_copy.cOPN)

# EXT_wordvec_df.shape, NEU_wordvec_df.shape, AGR_wordvec_df.shape, CON_wordvec_df.shape, OPN_wordvec_df.shape


# %%
# EXT_wordvec_df[EXT_wordvec_df.cEXT == 'y'].head(10)


# %%
# label = { 'y': 1 , 'n': 0}
# # EXT_wordvec_df.cEXT = [label[item] for item in EXT_wordvec_df.cEXT]
# EXT_wordvec_df[20:30]


# %%
# # Exporting to arff (weka file extension)
# arff.dump('ext_we_dataset_GP.arff'
#       , EXT_wordvec_df.values
#       , relation='extroversionGP'
#       , names=EXT_wordvec_df.columns)


# %%
