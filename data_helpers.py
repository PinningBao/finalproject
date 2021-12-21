import numpy as np
import re
import itertools
import codecs
from collections import Counter
import jieba

def clean_str(string):
  #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  return re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string).strip().lower()


def load_data_and_labels(pos=None,neg=None):
  positive=[[item for item in jieba.cut(s, cut_all=False)] for s in list(codecs.open("./data/chinese/pos.txt", "r", "utf-8").readlines())]
  negative=[[item for item in jieba.cut(s, cut_all=False)] for s in list(codecs.open("./data/chinese/neg.txt", "r", "utf-8").readlines())]
  x_text=positive+negative
  positive_labels = [[0, 1] for _ in positive]
  negative_labels = [[1, 0] for _ in negative]
  y = np.concatenate([positive_labels, negative_labels], 0)
  return [x_text,y]




def load_test_data_and_labels(pos=None,neg=None):

  positive=[[item for item in jieba.cut(s, cut_all=False)] for s in list(codecs.open("./data/test_text/pos.txt", "r", "utf-8").readlines())]
  negative=[[item for item in jieba.cut(s, cut_all=False)] for s in list(codecs.open("./data/test_text/neg.txt", "r", "utf-8").readlines())]
  positive_labels = [[0, 1] for _ in positive]
  negative_labels = [[1, 0] for _ in negative]
  y = np.concatenate([positive_labels, negative_labels], 0)
  return [x_text,y]


def pad_sentences(sentences, padding_word="<PAD/>"):
 
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = np.array(labels)
  return [x, y]


def load_data():
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, labels = load_data_and_labels()
  sentences_padded = pad_sentences(sentences)
  vocabulary, vocabulary_inv = build_vocab(sentences_padded)
  x, y = build_input_data(sentences_padded, labels, vocabulary)
  return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]# shuffled_data按照上述乱序得到新的样本
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):#开始生成batch
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)#这里主要是最后一个batch可能不足batchsize的处理
            yield shuffled_data[start_index:end_index]
            #yield，在for循环执行时，每次返回一个batch的data，占用的内存为常数
