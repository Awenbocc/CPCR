

import torch.nn as nn
from .connect import FCNet, BCNet
from torch.nn.utils.weight_norm import weight_norm

# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()
        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

class BiResNet(nn.Module):
    def __init__(self,args,dataset,glimpse):
        super(BiResNet,self).__init__()

        # # init Bilinear residual network
        self.glimpse = glimpse
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        for i in range(self.glimpse):
            b_net.append(BCNet(dataset.v_dim, args.hid_dim, args.hid_dim, None, k=1))
            q_prj.append(FCNet([args.hid_dim, args.hid_dim], '', .2))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.args = args

    def forward(self, v_emb, q_emb,att_p):
        b_emb = [0] * self.glimpse
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)

class BAN(nn.Module):
    def __init__(self, args, dataset):
        super(BAN,self).__init__()
        self.glimpse = args.glimpse
        self.bi_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, self.glimpse)
        self.bi_resnet = BiResNet(args,dataset, self.glimpse)

    def forward(self, v_emb, q_emb):
        att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
            # bilinear residual network
        last_output = self.bi_resnet(v_emb,q_emb,att_p)

        return last_output
        