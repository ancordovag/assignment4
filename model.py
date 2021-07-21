import sys

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
        self.pretrained_emb = pretrained_emb
        self.conv_dim = configs["conv_dim"]
        self.embsize = pretrained_emb.shape[-1]
        self.embed = nn.Embedding(token_size,self.embsize).from_pretrained(pretrained_emb)
        self.conv = nn.Conv1d(in_channels=self.embsize, out_channels=self.conv_dim,
                              kernel_size=3,stride=1,padding=1,padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.conv_dim,out_channels=self.conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        self.relu = F.relu


    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return
        the sentence embeddings
        """
        #print("Shape Pretrained emb: {}".format(self.pretrained_emb.shape)) #(16690,300)
        #print("Shape sent {}".format(sent.shape)) #(16,50)
        x = self.embed(sent).permute(0, 2, 1)
        #print("Shape embeded sent {}".format(x.shape)) #(16,300,50)
        c1 = self.relu(self.conv(x))
        #print("c1", c1.shape) #(16,25,50)
        c2 = self.relu(self.conv2(c1))
        #print("c2", c2.shape) #(16,25,50)
        u1 = torch.max(c1,2)[0]
        u2 = torch.max(c2,2)[0]
        #print("u1,u2 shape:", u1.shape, u2.shape) #(16,25) x2
        sent_embs = torch.cat([u1,u2],1)
        #print("Sent Embs", sent_embs.shape) # (16,50)

        return sent_embs


class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(NLINet, self).__init__()
        """
        Fill this method. You can define the FFNN, dropout and the sentence
        encoder here.
        """
        self.hidden = configs["mlp_hidden"]
        self.conv_dim = configs["conv_dim"]
        self.encoder = SentEncoder(configs, pretrained_emb, token_size, label_size)
        self.linear1 = nn.Linear(8*self.conv_dim,self.hidden)
        self.linear2 = nn.Linear(self.hidden, label_size)
        self.dropout = nn.Dropout(p=0.5)
        self.soft = nn.Softmax(-1)

    def forward(self, premise, hypothesis):
        """
        Fill this method. It should accept a pair of sentence (premise &
        hypothesis) and return the logits.
        """
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        concat = torch.cat([u,v],-1)
        #print("Concat shape: {}".format(concat.shape)) #(16,100)
        z = torch.cat([concat,torch.abs(u-v),u*v],dim=-1)
        #print("U, V und Z: {}, {}, {}".format(u.size(),v.size(),z.size())) # (16,50) (16,50) (16,200)
        #z = z.view(16,-1)
        x = self.linear1(self.dropout(z))
        x = self.linear2(x)
        out = self.soft(x)

        return out
