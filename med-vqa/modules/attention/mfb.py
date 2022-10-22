# -*- coding: utf-8 -*-#

# --------------------------------------------------------
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------
import torch.nn.functional as F
import torch.nn as nn
import torch


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class MFB(nn.Module):
    def __init__(self, args, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.args = args
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, args.MFB_K * args.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, args.MFB_K * args.MFB_O)
        self.dropout = nn.Dropout(args.DROPOUT_R)
        self.pool = nn.AvgPool1d(args.MFB_K, stride=args.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.args.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.args.MFB_O)      # (N, C, O)
        return z, exp_out


class QAtt(nn.Module):
    def __init__(self, args):
        super(QAtt, self).__init__()
        self.args = args
        self.mlp = MLP(
            in_size=args.LSTM_OUT_SIZE,
            mid_size=args.HIDDEN_SIZE,
            out_size=args.Q_GLIMPSES,
            dropout_r=args.DROPOUT_R,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)

        qatt_feat_list = []
        for i in range(self.args.Q_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, args, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.DROPOUT_R)
        self.mfb = MFB(args, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=args.MFB_O,
            mid_size=args.HIDDEN_SIZE,
            out_size=args.I_GLIMPSES,
            dropout_r=args.DROPOUT_R,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)

        iatt_feat_list = []
        for i in range(self.args.I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self, args):
        super(CoAtt, self).__init__()
        self.args = args

        img_feat_size = 128
        img_att_feat_size = img_feat_size * args.I_GLIMPSES #256
        ques_att_feat_size = args.LSTM_OUT_SIZE * args.Q_GLIMPSES # 2048

        self.q_att = QAtt(args)
        self.i_att = IAtt(args, img_feat_size, ques_att_feat_size)

        if self.args.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(args, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(args, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(args, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''

        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        if self.args.HIGH_ORDER:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z

