import os
import argparse
import yaml
import random
import torch
import time

from utils import Dataset
from model import SentEncoder
from model import NLINet
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn as nn
from tqdm import tqdm
from time import sleep


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--RUN_MODE', dest='RUN_MODE',
        choices=['train', 'val', 'test'],
        help='{train, val, test}',
        type=str, required=True
    )

    parser.add_argument(
        '--CPU', dest='CPU',
        help='use CPU instead of GPU',
        action='store_true'
    )

    parser.add_argument(
        '--RESUME', dest='RESUME',
        help='resume training',
        action='store_true'
    )

    parser.add_argument(
        '--CKPT_E', dest='CKPT_EPOCH',
        help='checkpoint epoch',
        type=int
    )

    parser.add_argument(
        '--VERSION', dest='VERSION',
        help='model version',
        type=int
    )

    parser.add_argument(
        '--DEBUG', dest='DEBUG',
        help='enter debug mode',
        action='store_true'
    )

    args = parser.parse_args()
    return args

class MainExec(object):
    def __init__(self, args, configs):
        self.args = args
        self.cfgs = configs

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # for failsafe

        if self.args.VERSION is None:
            self.model_ver = str(random.randint(0, 99999999))
        else:
            self.model_ver = str(self.args.VERSION)

        print("Model version:", self.model_ver)

        # Fix seed
        self.seed = int(self.model_ver)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)

    def train(self):
        data = Dataset(self.args)
        pretrained_emb = data.pretrained_emb
        token_size = data.token_size
        label_size = data.label_size
        data_size = data.data_size
        #print("Token Size = {}".format(token_size))
        #print("Label Size = {}".format(label_size))
        #print("Pretrained Embed Size = {}".format(pretrained_emb.size))

        """
        # TODO: You should declare the model here (and send it to your selected device).
        You should define the loss function and optimizer, with learning
        rate obtained from the configuration file. You should also use
        `torch.utils.data.Dataloader` to load the data from Dataset object.
        For more information, see:
        https://pytorch.org/docs/stable/data.html#module-torch.utils.data .
        """
        pretrained_emb_torch = torch.from_numpy(pretrained_emb)
        batch_size = self.cfgs["batch_size"]
        lr = self.cfgs["lr"]

        net = NLINet(configs=self.cfgs,pretrained_emb=pretrained_emb_torch,token_size=token_size,label_size=label_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(net.parameters(), lr=lr)

        dataloader = DataLoader(data,batch_size=batch_size)

        # -----------------------------------------------------------------------

        if self.args.RESUME:
            print('Resume training...')
            start_epoch = self.args.CKPT_EPOCH
            path = os.path.join(os.getcwd(),
                                self.model_ver,
                                'epoch' + str(start_epoch) + '.pkl')

            # Load state dict of the model and optimizer
            ckpt = torch.load(path, map_location=self.device)
            net.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            start_epoch = 0
            os.mkdir(os.path.join(os.getcwd(), self.model_ver))

        loss_sum = 0

        for epoch in range(start_epoch, self.cfgs["epochs"]):
            with tqdm(dataloader) as tepoch:
                for step, (
                    premise_iter,
                    hypothesis_iter,
                    label_iter
                ) in enumerate(tepoch):
                    tepoch.set_description("Epoch {}".format(str(epoch)))
                    """
                    #TODO: Fill the training loop.
                    """
                    optimizer.zero_grad()
                    outputs = net(premise_iter,hypothesis_iter)
                    print("Outputs: {}".format(outputs))
                    print("Label: {}".format(label_iter))
                    loss = loss_fn(outputs, label_iter)
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()
                    # ---------------------------------------------------
                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)

            print('Average loss: {:.4f}'.format(loss_sum/len(dataloader)))
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(
                state,
                os.path.join(os.getcwd(),
                             self.model_ver,
                             'epoch' + str(epoch_finish) + '.pkl')
            )

            loss_sum = 0

    def eval(self):
        data = Dataset(self.args)
        pretrained_emb = data.pretrained_emb
        token_size = data.token_size
        label_size = data.label_size
        data_size = data.data_size

        """
        # TODO: You should declare the model here (and send it to your selected device).
        Don't forget to set the model to evaluation mode. You should also use
        `torch.utils.data.Dataloader` to load the data from Dataset object.
        """
        pretrained_emb_torch = torch.from_numpy(pretrained_emb)
        batch_size = self.cfgs["batch_size"]

        net = NLINet(configs=self.cfgs, pretrained_emb=pretrained_emb_torch, token_size=token_size, label_size=label_size)
        path = os.path.join(os.getcwd(),
                            self.model_ver,
                            'epoch' + str(self.args.CKPT_EPOCH) + '.pkl')
        dataloader = DataLoader(data, batch_size=batch_size)

        # Load state dict of the model
        ckpt = torch.load(path, map_location=self.device)
        net.load_state_dict(ckpt['state_dict'])
        net.eval()

        """TODO : Evaluate the model using accuracy as metrics."""
        corrects = 0
        N = 0
        with tqdm(dataloader) as tepoch:
            for step, (
                    premise_iter,
                    hypothesis_iter,
                    label_iter
            ) in enumerate(tepoch):
                outputs = net(premise_iter, hypothesis_iter)
                for l, o in zip(label_iter, outputs):
                    if l == o:
                        corrects += 1
                N += batch_size
        accuracy = corrects / N
        print(accuracy)

        # -------------------------------------------------

    def overfit(self):
        data = Dataset(self.args)
        pretrained_emb = data.pretrained_emb
        token_size = data.token_size
        label_size = data.label_size
        data_size = data.data_size

        """
        TODO : You should declare the model here (and send it to your selected device).
        You should define the loss function and optimizer, with learning
        rate obtained from the configuration file. You should also use
        `torch.utils.data.Dataloader` to load the data from Dataset object.
        Use only a single batch to ensure your model is working correctly.
        """
        pretrained_emb_torch = torch.from_numpy(pretrained_emb)
        batch_size = 1
        lr = self.cfgs["lr"]

        net = NLINet(configs=self.cfgs, pretrained_emb=pretrained_emb_torch, token_size=token_size, label_size=label_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(net.parameters(), lr=lr)
        dataloader = DataLoader(data, batch_size=batch_size)

        # -----------------------------------------------------------------------
        start_epoch = 0

        """
        TODO : Train using a single batch and observe the loss. Does it converge?.
        """
        loss_sum = 0
        with tqdm(dataloader) as tepoch:
            for step, (
                    premise_iter,
                    hypothesis_iter,
                    label_iter
            ) in enumerate(tepoch):
                optimizer.zero_grad()
                outputs = net(premise_iter, hypothesis_iter)
                loss = loss_fn(outputs, label_iter)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                print(loss_sum)

        # -----------------------------------------------------------------

    def run(self, run_mode):
        if run_mode == 'train' and self.args.DEBUG:
            print('Overfitting a single batch...')
            self.overfit()
        elif run_mode == 'train':
            print('Starting training mode...')
            self.train()
        elif run_mode == 'val':
            print('Starting validation mode...')
            self.eval()
        elif run_mode == 'test':
            print('Starting test mode...')
            self.eval()
        else:
            exit(-1)


if __name__ == "__main__":
    args = parse_args()

    with open('./config.yml', 'r') as f:
        model_config = yaml.safe_load(f)

    exec = MainExec(args, model_config)
    exec.run(args.RUN_MODE)
