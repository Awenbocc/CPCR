# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/4/8
#-------------------------------------------------------------------------------
import os
from statistics import mode
import time
import torch
from modules import utils
from datetime import datetime
import torch.nn as nn

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp
    
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_score_with_logits_ce(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    scores = sum(logits == labels)
    return scores

# Train phase
def train(args, model, train_loader, eval_loader,s_opt=None, s_epoch=0):
    device = args.device
    model = model.to(device)
    # create packet for output
    utils.create_dir(args.output)
    # for every train, create a packet for saving .pth and .log
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output,run_timestamp)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    # Adamax optimizer
    optim = torch.optim.Adamax(params=model.parameters())
    # Scheduler learning rate
    #lr_decay = lr_scheduler.CosineAnnealingLR(optim,T_max=len(train_loader))  # only fit for sgdr


    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss() if not args.tcr else torch.nn.CrossEntropyLoss()
    ae_criterion = torch.nn.MSELoss()

    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase
    for epoch in range(s_epoch, args.epochs):
        total_loss = 0
        train_score = 0
        item = 0
        model.train()
        start= get_time_stamp()
        # Predicting and computing score
        for i, (v, q, a, answer_type, question_type, answer_target) in enumerate(train_loader):
            #lr_decay.step()
            open_idx = (answer_target == 1).nonzero().flatten()
            close_idx = (answer_target == 0).nonzero().flatten()
            optim.zero_grad()
            
            v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            v[0] = v[0].to(device)

            v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[1] = v[1].to(device)
            
            
            
            q = q.to(device)
            a = a.to(device)
            open_idx = open_idx.to(device)
            close_idx = close_idx.to(device)
            # MEVF loss computation
            if args.autoencoder:
                features, decoder = model(v, q)
            else:
                if not args.tcr:
                    features = model(v, q)
                else:
                    features = model(v, q, open_idx, close_idx)

            if not args.tcr:
                preds = model.classify(features)
                loss = criterion(preds.float(), a)
            else:
                preds_open, preds_close = model.classify(features)
                a_open, a_close = a[open_idx], a[close_idx]
                loss_open = criterion(preds_open.float(), a_open)
                loss_close = criterion(preds_close.float(), a_close)
                loss = loss_open + loss_close

            if args.autoencoder:
                loss_ae = ae_criterion(v[1], decoder)
                loss = loss + (loss_ae * args.ae_alpha)
            # loss /= answers.size()[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.25)
            optim.step()

            total_loss += loss.item()
            item += len(answer_target)
            if not args.tcr:
                final_preds = preds
                batch_score = compute_score_with_logits(final_preds, a.data).sum()
                train_score += batch_score
            else:
                open_score = compute_score_with_logits_ce(preds_open, a_open.data)
                close_score = compute_score_with_logits_ce(preds_close, a_close.data)
                train_score += open_score + close_score



        end = get_time_stamp()
        print('train',start,end)
        total_loss /= len(train_loader)
        train_score = 100 * train_score / item
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        # Evaluation
        if eval_loader is not None:
            
            eval_score = evaluate(model, eval_loader, args,logger)
            
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                # Save the best acc epoch
                model_path = os.path.join(ckpt_path, '{}.pth'.format(best_epoch))
                # utils.save_model(model_path, model, best_epoch)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))

        
        
# Evaluation
def evaluate(model, dataloader, args,logger):
    device = args.device
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    test_start = get_time_stamp()
    with torch.no_grad():
        for i,(v, q, a,answer_type, question_type, answer_target) in enumerate(dataloader):
            #if i==1:
             #   break
            open_idx = (answer_target == 1).nonzero().flatten()
            close_idx = (answer_target == 0).nonzero().flatten()
            v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            v[0] = v[0].to(device)

            v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[1] = v[1].to(device)

            q = q.to(device)
            a = a.to(device)
            open_idx = open_idx.to(device)
            close_idx = close_idx.to(device)

            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)

            size = q.shape[0]

            if not args.tcr:
                
                preds = model.classifier(features)
                final_preds = preds
                batch_score = compute_score_with_logits(final_preds, a.data)
                score += batch_score.sum()
                # open-ended
                
                for j in range(size):
                    if answer_type[j]=='OPEN':
                        open_ended += 1
                        score_open += 1 if batch_score[j].sum() == 1 else 0
                    # closed-ended
                    else:
                        closed_ended += 1
                        score_close += 1 if batch_score[j].sum() == 1 else 0
            else:
                features = model(v, q, open_idx, close_idx)
                preds_open, preds_close = model.classify(features)
                open_ended += len(open_idx)
                closed_ended += len(close_idx)
                batch_score_open = compute_score_with_logits_ce(preds_open, a[open_idx].data)
                batch_score_close = compute_score_with_logits_ce(preds_close, a[close_idx].data)
                score += batch_score_open + batch_score_close
                score_open += batch_score_open
                score_close += batch_score_close

            total += size
                       
            


    test_end = get_time_stamp()
    print('predict:',score, score_open, score_close)
    print('dataset:',total, open_ended, closed_ended)
#     print('test',test_start,test_end)
    score = 100* score / total
    open_score = 100* score_open/ open_ended if not open_ended == 0 else 0
    close_score = 100* score_close/ closed_ended if not closed_ended == 0 else 0
    
    
    logger.info('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%' .format(score,open_score,close_score))
    return score
