#coding:utf-8
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors,LineSentence
import itertools
import os
from data import get_vocab
MAX_WORDS_IN_BATCH=1000


logger=logging.Logger(name="word2vec",level=logging.INFO)
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
logging.root.setLevel(level=logging.INFO)

train_triple_path="./datasets/train_triples/seg_train_triples.txt"
test_triple2016_path="./datasets/train_triples/seg_test_triples.txt"
test_triple2017_path="./datasets/train_triples/seg_test_triples2017.txt"
kb_path="./datasets/kb/segmented_kbqa.kb"

model_path="./datasets/temp/word2vec.bin"
emb_dim=128

def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)
to_unicode = any2unicode
class MyLineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, sources, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).
        """
        self.sources = sources
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        for source in self.sources:
            try:
                # Assume it is a file-like object and try treating it as such
                # Things that don't have seek will trigger an exception
                source.seek(0)
                for line in itertools.islice(source, self.limit):
                    line = to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length
            except AttributeError:
                # If it didn't work like a file, use it as a string filename
                with open(source,encoding="utf-8") as fin:
                    for line in itertools.islice(fin, self.limit):
                        line = to_unicode(line).split()
                        i = 0
                        while i < len(line):
                            yield line[i : i + self.max_sentence_length]
                            i += self.max_sentence_length
                
def train_word2vec():
    '''训练词项向量
    '''
    model=Word2Vec(sentences=MyLineSentence([kb_path,train_triple_path,test_triple2016_path,test_triple2017_path]),size=emb_dim,window=5,min_count=5,iter=5)
    model.wv.save_word2vec_format(model_path,binary=True)
    return model
    
def get_embedding():
    emb_path="datasets/temp/embedding.np"
    if os.path.exists(emb_path):
        return np.load(open(emb_path,'rb'))
    else:
        model=KeyedVectors.load_word2vec_format(model_path,binary=True)
        iw,vocab=get_vocab()
        size=len(list(vocab.keys()))
        emb=np.zeros(shape=[size,emb_dim])
        for word,index in vocab.items():
            if index in [0,1]:
                continue
            emb[index]=model[word]
        np.save(open(emb_path,"wb"),emb)
        return emb
    
if __name__=="__main__":
    train_word2vec()
    model=KeyedVectors.load_word2vec_format(model_path,binary=True)