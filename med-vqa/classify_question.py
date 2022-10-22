# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         classify_question
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/5/14
#-------------------------------------------------------------------------------


import torch
from modules.vqa_data import Dictionary, get_dataset
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from modules.language_model import WordEmbedding,QuestionEmbedding
import argparse
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from modules import utils
from datetime import datetime
import random
import numpy as np
import math

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin



# change the question b*number*hidden -> b*hidden
class QuestionAttention(nn.Module):
    def __init__(self, dim, context_dim=300):
        super().__init__()

        self.tanh_gate = linear(context_dim + dim, dim)
        self.sigmoid_gate = linear(context_dim + dim, dim)
        self.attn = linear(dim, 1)
        self.dim = dim

    def forward(self, context, question):  #b*12*600 b*12*1024

        concated = torch.cat([context, question], -1)  #b*12*600 + 1024
        concated = torch.mul(torch.tanh(self.tanh_gate(concated)), torch.sigmoid(self.sigmoid_gate(concated)))  #b*12*1024
        a = self.attn(concated) # #b*12*1
        attn = F.softmax(a.squeeze(), 1) #b*12

        ques_attn = torch.bmm(attn.unsqueeze(1), question).squeeze() #b*1024

        return ques_attn

class SimpleQCR(nn.Module):
    def __init__(self, dim, context_dim):
        super(SimpleQCR, self).__init__()
        self.q_attn = QuestionAttention(dim, context_dim)
        self.proj = linear(dim, dim)
    
    def forward(self, w_emb, q_emb):

        q_att = self.q_attn(w_emb, q_emb)
        return self.proj(q_att)


class QCR(nn.Module):
    def __init__(self, size_question, path_init):
        super(QCR, self).__init__()
        self.w_emb = WordEmbedding(size_question, 300, 0.0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 1024, 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024, 1024)
        self.f_fc2 = linear(1024, 1024)
        self.f_fc3 = linear(1024, 1024)

    def forward(self, question):
        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_final = self.q_final(w_emb, q_emb)  # b, 1024

        x_f = self.f_fc1(q_final)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f

class Classify_Model(nn.Module):
    def __init__(self,size_question,path_init):
        super(Classify_Model,self).__init__()
        self.w_emb = WordEmbedding(size_question,300, .0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 512 , 1, False, .0, 'GRU')
        self.q_final = QuestionAttention(512)
        self.f_fc1 = linear(512,64)
        # self.f_fc2 = linear(256,64)
        self.f_fc3 = linear(64,2)
        self.drouput = nn.Dropout(0.5)



    def forward(self,question):

        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_final = self.q_final(w_emb,q_emb) #b, 1024

        x_f = self.f_fc1(q_final)
        x_f = self.drouput(F.relu(x_f))

        # x_f = self.f_fc2(x_f)
        # x_f = self.drouput(F.relu(x_f))

        x_f = self.f_fc3(x_f)

        return x_f


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA over MAC")
    # GPU config
    parser.add_argument('--seed', type=int, default=105
                        , help='random seed for gpu.default:5')
    parser.add_argument('--question_len', type=int, default=12)
    parser.add_argument('--dataset', type=str, default='rad')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--v_dim', type=int, default=32)
    parser.add_argument('--tcr', default=False)

    args = parser.parse_args()
    return args


# Evaluation
def evaluate(model, dataloader,logger,device):
    score = 0
    number =0
    model.eval()
    ids = []
    with torch.no_grad():
        for i,row in enumerate(dataloader):
            id, image_data, question, target, answer_type, question_type, answer_target = row
            question, answer_target = question.to(device), answer_target.to(device)
            output = model(question)
            pred = output.data.max(1)[1]
            correct = pred.eq(answer_target.data).cpu().sum()
            
            ids.extend(id[(pred!=answer_target.data).cpu().nonzero().flatten()])


            score += correct.item()
            number += len(answer_target)

        score = score / number * 100.
        print(ids)
    logger.info('[Validate] Val_Acc:{:.6f}%'.format(score))
    return score

def adjust_lr(optimizer, lr, epoch, epochs, schedule):

    # lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        

if __name__=='__main__':
    dataroot = './data/data_RAD'
    args = parse_args()

    # # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = get_dataset(args,'rad','train')

    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    val_dataset = get_dataset(args,'rad','test')
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = Classify_Model(train_dataset.dictionary.ntoken,'./data/rad/glove6b_init_300d.npy')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_params1 = sum(p.numel() for p in model.q_final.parameters())
    total_params2 = sum(p.numel() for p in model.f_fc1.parameters())
    total_params3 = sum(p.numel() for p in model.q_emb.parameters())
    total_params4 = sum(p.numel() for p in model.f_fc3.parameters())
    print('parameters:',  (total_params1+total_params2+total_params3+total_params4)/1000000.)
    print('parameters:',  total_params3/1000000.)
    print('parameters:',  total_params/1000000.)

    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    tips = f'{run_timestamp}_TCR'
    ckpt_path = os.path.join('./saved_models', tips)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'TCR.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())

    #
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    epochs = 150
    adjust_seq = [40,120]
    best_eval_score = -9999
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        acc = 0.
        number_dataset = 0
        total_loss = 0

    
        # adjust_lr(optimizer,lr,epoch,epochs,adjust_seq)


        for i, row in enumerate(train_data):
            id, image_data, question, target, answer_type, question_type, answer_target = row

            question, answer_target = question.to(device), answer_target.to(device)

            optimizer.zero_grad()


            output = model(question)

            loss = criterion(output,answer_target)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(),0.25)
            optimizer.step()

            pred = output.data.max(1)[1]

            correct = (pred==answer_target).data.cpu().sum()
            
            acc += correct.item()
            number_dataset += len(answer_target)
            total_loss+= loss
        

        total_loss /= len(train_data)
        acc = acc/ number_dataset * 100.

        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, acc
                                                                     ))
        # Evaluation
        if val_data is not None:
            eval_score = evaluate(model, val_data, logger, device)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                utils.save_model(ckpt_path+'/best.pth', model, epoch)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))



        









