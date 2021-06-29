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
        conv_dim = configs["conv_dim"]
        self.embed = nn.Embedding.from_pretrained(pretrained_emb)
        self.conv = nn.Conv1d(50, conv_dim,(2))
        self.relu = nn.ReLU


    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return
        the sentence embeddings
        """
        x = self.embed(sent)
        x = self.conv(x)
        sent_embs = self.relu(x)

        return sent_embs


class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(NLINet, self).__init__()
        """
        Fill this method. You can define the FFNN, dropout and the sentence
        encoder here.
        """
        hidden = configs["mlp_hidden"]
        self.encoder = SentEncoder(configs, pretrained_emb, token_size, label_size)
        self.linear = nn.Linear(hidden, label_size)
        self.dropbox = nn.Dropout(p=0.3)
        self.soft = nn.Softmax()

    def forward(self, premise, hypothesis):
        """
        Fill this method. It should accept a pair of sentence (premise &
        hypothesis) and return the logits.
        """
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        z = torch.cat([u,v,torch.abs(u-y),u*v],dim=0)
        x = self.linear(z)
        x = self.dropout(x)
        out = self.soft(x)

        return out
