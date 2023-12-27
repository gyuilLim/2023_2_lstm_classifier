# tokenized_sentences 받아서 vectorize
from utils.tokenizer import Vocabulary
import numpy as np

class Vectorize() :
    def __init__(self, tokenized_sents, vocab, max_length) :
        self._tokenized_sents = tokenized_sents
        self._vocab = vocab
        self.vocabulary = Vocabulary(self._vocab)
        self.max_length = max_length
        
    def vectorizer(self) :
        out_vector = []
        for sentence in self._tokenized_sents :
            indices = [self.vocabulary.lookup_token(token) for token in sentence]

            
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]

            temp_vector = np.zeros(self.max_length, dtype=np.int64)
            temp_vector[-len(indices):] = indices
            temp_vector[:len(indices)] = 0
            
            out_vector.append(temp_vector)

        return out_vector, len(indices)