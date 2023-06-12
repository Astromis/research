#https://github.com/mttk/rnn-classifier/blob/master/model.py

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

RNNS = ['LSTM', 'GRU']

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
                            dropout=dropout, bidirectional=bidirectional,)# batch_first=True

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

class SiamesRNN(nn.Module):
    def __init__(self, embedding, encoder, attention, 
                 hidden_dim, intermidiate_dim, num_classes):
        super(SiamesRNN, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, intermidiate_dim)
        self.classifier_head = nn.Linear(intermidiate_dim*2, num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))


    def forward(self, input):
        #print("Input", input.shape)
        left = input["premise"]
        right = input["hypothesis"]
        outputs_l, hidden_l = self.encoder(self.embedding(left))
        if isinstance(hidden_l, tuple): # LSTM
            hidden_l = hidden_l[1] # take the cell state
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden_l = torch.cat([hidden_l[-1], hidden_l[-2]], dim=1)
        else:
            hidden_l = hidden_l[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

        energy, linear_combination_l = self.attention(hidden_l, outputs_l, outputs_l) 
        #print("energy", energy.shape)
        #print("linear_combination", linear_combination.shape)
        logits_l = self.decoder(linear_combination_l)
        #print("logits", logits.shape)
        
        outputs_r, hidden_r = self.encoder(self.embedding(right))
        if isinstance(hidden_r, tuple): # LSTM
            hidden_r = hidden_r[1] # take the cell state
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden_r = torch.cat([hidden_r[-1], hidden_r[-2]], dim=1)
        else:
            hidden_r = hidden_r[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

        energy, linear_combination_r = self.attention(hidden_r, outputs_r, outputs_r) 

        logits_r = self.decoder(linear_combination_r)
        
        intermidiate = torch.cat([logits_l, logits_r], dim=1)
        final_logits = self.classifier_head(intermidiate)
        
        return final_logits

class RNN(nn.Module):
    def __init__(self, embedding, encoder, attention, hidden_dim, num_classes):
        super(RNN, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))


    def forward(self, input):
        #print("Input", input.shape)
        outputs, hidden = self.encoder(self.embedding(input["text"]))
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state
        #print("outputs", outputs.shape)
        #print("hidden", hidden.shape)
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

        energy, linear_combination = self.attention(hidden, outputs, outputs) 
        #print("energy", energy.shape)
        #print("linear_combination", linear_combination.shape)
        logits = self.decoder(linear_combination)
        #print("logits", logits.shape)
        return logits#, energy
    

class SentenceBERTClassifier(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    """
    def __init__(self,
                 model,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        super().__init__()
        self.sent_bert_model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)

    def forward(self, input: List[Tuple[str, str]]):
        rep_a = self.sent_bert_model.encode(input[0], convert_to_tensor=True, convert_to_numpy=False ) 
        rep_b = self.sent_bert_model.encode(input[1], convert_to_tensor=True, convert_to_numpy=False )

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        return output