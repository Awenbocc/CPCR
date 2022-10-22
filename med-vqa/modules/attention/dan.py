import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class rDAN(nn.Module):
    def __init__(self, args, k=2):
        super(rDAN, self).__init__()

        self.args = args
        memory_size = args.hid_dim  # bidirectional

        # Visual Attention
        self.Wv = nn.Linear(in_features=128, out_features=args.hid_dim)
        self.Wvm = nn.Linear(in_features=memory_size, out_features=args.hid_dim)
        self.Wvh = nn.Linear(in_features=args.hid_dim, out_features=1)
        self.P = nn.Linear(in_features=128, out_features=args.hid_dim)

        # Textual Attention
        self.Wu = nn.Linear(in_features=args.hid_dim, out_features=args.hid_dim)
        self.Wum = nn.Linear(in_features=memory_size, out_features=args.hid_dim)
        self.Wuh = nn.Linear(in_features=args.hid_dim, out_features=1)

#         self.Wans = nn.Linear(in_features=memory_size, out_features=answer_size)

        # Scoring Network
        # self.classifier = Classifier(memory_size, hidden_size, answer_size, 0.5)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0)  # Softmax over first dimension

        # Loops
        self.k = k

    def forward(self, visual, text):
        batch_size = visual.shape[0]

        # Prepare Visual Features
        visual = visual.view(batch_size, 128, -1)
        vns = visual.permute(2, 0, 1)  # (nregion, batch_size, dim)

        # Prepare Textual Features
        text = text.permute(1, 0,2) # (seq_len, batch_size, dim)


        # Initialize Memory
        u = text.mean(0)                   # batch, hid_dim 
        v = self.tanh(self.P(vns.mean(0))) # 64 1024
        memory = v * u

        # K indicates the number of hops
        for k in range(self.k):
            # Compute Visual Attention 1 64 1024 
            hv = self.tanh(self.Wv(self.dropout(vns))) * self.tanh(self.Wvm(self.dropout(memory)))
            # attention weights for every region
            alphaV = self.softmax(self.Wvh(self.dropout(hv)))  # (seq_len, batch_size, memory_size)
            # Sum over regions
            v = self.tanh(self.P(alphaV * vns)).sum(0)

            # Text
            # (seq_len, batch_size, dim) * (batch_size, dim)
            hu = self.tanh(self.Wu(self.dropout(text))) * self.tanh(self.Wum(self.dropout(memory)))
            # attention weights for text features
            alphaU = self.softmax(self.Wuh(self.dropout(hu)))  # (seq_len, batch_size, memory_size)
            # Sum over sequence
            u = (alphaU * text).sum(0)  # Sum over sequence

            # Build Memory
            memory = memory + u * v

        # We compute scores using a classifier
        # scores = self.classifier(memory)

        return memory


if __name__ == "__main__":
    pass