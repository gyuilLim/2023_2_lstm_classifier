from collections import Counter
import nltk.data

class Tokenize() :

    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.vocab = {"<mask>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}

    def sent_tokenize(self, sentence, use_vocab=True):
            sentence = sentence.lower()
            tokenized = ['<s>'] + sentence.split() + ['</s>']
            if use_vocab:
                for i in range(len(tokenized)):
                    if tokenized[i] not in self.vocab:
                        tokenized[i] = list(self.vocab.keys())[0]

            return tokenized

    def doc_tokenize(self, comments, train):
            tokenized_sents = []
            if train:
                for sentence in comments:
                    tokenized = self.sent_tokenize(sentence, False)
                    tokenized_sents.append(tokenized)
                counter_dict = dict(Counter([y for x in tokenized_sents for y in x]))
                for keys in counter_dict.keys():
                    if keys not in self.vocab and counter_dict[keys] >= 50:
                        self.vocab[keys] = list(self.vocab.values())[-1]+1
                return self.doc_tokenize(comments, train=False)
            else:
                if len(self.vocab) <= 3:
                    print("Error Case - Train corpus should be processed in advance.")
                    exit()

                for sentence in comments:
                    tokenized_sents.append(self.sent_tokenize(sentence, use_vocab=True))
                
            return tokenized_sents, self.vocab
        
class Vocabulary() :
    def __init__(self, vocab) :
        self.token_to_idx = vocab
        self.idx_to_token = [{value : key} for key, value in vocab.items()]
    
    def lookup_token(self, token):
        if token not in self.token_to_idx :
            return self.token_to_idx["<unk>"]
        return self.token_to_idx[token]
    
    def lookup_index(self, idx):
        return self.idx_to_token[idx]
    
    def __len__(self):
        return len(self.token_to_idx)