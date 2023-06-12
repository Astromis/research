
from typing import List

from nltk import word_tokenize


class DictTokenizer:
    """The tokenizer that perworm "simple" tokenization with NLTK."""
    def __init__(self):
        self.word2idx = {}
        self.padding_idx = None
        self.vocab_size = None
        self.is_trained = False
        
    def train(self, texts: List[str]):
        """Train the tokenizer.

        Args:
            texts (List[str]): Texts for training.
        """
        words = []
        for t in texts:
            for w in word_tokenize(t.lower()):
                words.append(w)
        self.word2idx = {w:i for i,w in enumerate(set(words))}
        self.word2idx["<PAD>"] = len(set(words))
        self.word2idx["<UNK>"] = len(set(words)) + 1
        self.padding_idx = self.word2idx["<PAD>"]
        self.unk_idx = self.word2idx["<UNK>"]
        self.vocab_size = len(self.word2idx)
        self.is_trained = True
        
    
    def _alignment(self, seq, max_len):
        """ Perform alignment on sequence, i.e. truncation
        and padding according to max_len
        """
        seq = seq[:max_len]
        diff = len(seq) - max_len
        if diff < 0 :
            seq.extend([self.padding_idx] * (max_len - len(seq) ))
        return seq
    
    def encode(self, text, add_special_tokens=False, padding="max_length", max_length=150, truncation="longest_first" ):
        """Method that mimics BertTokenizer for a convinience."""
        if not self.is_trained:
            raise ValueError("The tokenizer has not been trained")
        encoded_text = []
        for w in word_tokenize(text.lower()):
            if w in self.word2idx:
                encoded_text.append(self.word2idx[w])
            else:
                encoded_text.append(self.unk_idx)
        encoded_text = self._alignment(encoded_text, max_length)
        return encoded_text