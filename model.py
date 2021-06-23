import torch
import torch.nn as nn
import torch.nn.functional as F


class SentEncoder(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(SentEncoder, self).__init__()
        """
        Fill this method. You can define and load the word embeddings here.
        You should define the convolution layer here, which use ReLU
        activation. Tips: You can use nn.Sequential to make your code cleaner.
        """

    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return
        the sentence embeddings
        """

        return sent_embs


class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(NLINet, self).__init__()
        """
        Fill this method. You can define the FFNN, dropout and the sentence
        encoder here.
        """

    def forward(self, premise, hypothesis):
        """
        Fill this method. It should accept a pair of sentence (premise &
        hypothesis) and return the logits.
        """

        return out
