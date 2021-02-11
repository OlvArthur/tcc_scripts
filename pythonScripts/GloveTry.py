from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import preprocessing

glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


print('arquivo ${word2vec_output_file} criado')
model = KeyedVectors.load_word2vec_format('glove_s100.txt', binary=False)


# print(model.similarity('cachorro', 'gato'))
# model.most_similar(positive=['mulher', 'rei'], negative=['homen'], topn=10)
