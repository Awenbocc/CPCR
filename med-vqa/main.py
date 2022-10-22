# -*- coding: utf-8 -*-#

import argparse
from modules.vqa_data import get_dataset
import os
from torch.utils.data import DataLoader
from models.vqa_model import VQA_Model as Model 
import torch
from train import train
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # GPU config
    parser.add_argument('--seed', type=int, default=88
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models',
                        help='save file directory')
    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.005, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # Dataset Usages
    parser.add_argument('--dataset', type=str, choices=['rad','slake'])
    parser.add_argument('--use_data', action='store_true', default=True,
                         help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str, help='RAD dir')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu','sigmoid'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Attention --------------------------------------------------------------------------------------------------------
    # Choices of attention models
    parser.add_argument('-a','--attention', type=str, default='BAN', choices=['BAN','SAN','MFB','MFBCo','DAN','MFH','QO','VO'],
                        help='the model we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--glimpse', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--glimpse_open', type=int, default=1,
                        help='glimpse_open in Bilinear Attention Networks of TCR')
    parser.add_argument('--glimpse_close', type=int, default=1,
                        help='glimpse_close in Bilinear Attention Networks of TCR')  
    parser.add_argument('--tcr', action='store_true',help='use TCR module')
    parser.add_argument('--qcr', action='store_true',help='use QCR module')  


    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Question ---------------------------------------------------------------------------------------------------------
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--cat', type=bool, default=True,
                        help='concatenated 600-D word embedding')
    parser.add_argument('--hid_dim', type=int, default=1024,
                        help='dim of joint semantic features')

    # Vision -----------------------------------------------------------------------------------------------------------
    # Input visual feature dimension
    parser.add_argument('--v_dim', default=128, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    
    # other model hyper-parameters
    parser.add_argument('--other_model', action='store_true', help='End to end model?')
    parser.add_argument('--visual_extractor', type=str, default='resnet50')
    parser.add_argument('--arch', type=str, help='ckpt for other model')
    
    # details
    parser.add_argument('--details',type=str,default='original ')



    args = parser.parse_args()
    return args

if __name__ == '__main__':

    
    args = parse_args()
    args.root = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = args.root + '/data/' + args.dataset

    # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create word dictionary from train+val dataset
    # prepare the dataloader
    train_dataset = get_dataset(args, 'rad', 'train')
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2,drop_last=False,
                              pin_memory=True)

    val_dataset = get_dataset(args, 'rad', 'test')
    val_loader = DataLoader(val_dataset, args.batch_size , shuffle=False, num_workers=2, drop_last=False,
                              pin_memory=True)
    # create VQA model
    model = Model(train_dataset, args)
    total_params2 = sum(p.numel() for p in model.parameters())
    # total_params1 = sum(p.numel() for p in model.other.parameters())
    # total_params3 = sum(p.numel() for p in model.final.parameters())
    # total_params4 = sum(p.numel() for p in model.ae.parameters())
    # total_params5 = sum(p.numel() for p in model.convert.parameters())
#     print(total_params1)
    print('parameters:',  (total_params2)/1000000.)
    
    # # load snapshot
    # if args.input is not None:
    #     print('loading %s' % args.input)
    #     pre_ckpt = torch.load(args.input)
    #     model.load_state_dict(pre_ckpt.get('model_state', pre_ckpt))
    #     optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
    #     optim.load_state_dict(pre_ckpt.get('optimizer_state', pre_ckpt))
    #     epoch = pre_ckpt['epoch'] + 1
    #
    # # training phase
    train(args, model, train_loader, val_loader)

