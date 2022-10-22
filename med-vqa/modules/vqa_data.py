# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/5/1
#-------------------------------------------------------------------------------


"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
from modules import utils
import torch
from torch.utils.data import Dataset,DataLoader
import itertools
import warnings
from tools.create_dictionary import Dictionary
from PIL import Image
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

def _create_entry(img_id, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img_id,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_type' : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type' : data['phrase_type']}
    return entry

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(config, split, img2id):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    name: 'train', 'val', 'test'
    """
    samples = json.load(open(config[f'{split}_sample'],'r'))
    samples = sorted(samples, key=lambda x: x['qid'])

    answers = cPickle.load(open(config[f'{split}_target'], 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        entries.append(_create_entry(img2id[img_id], sample, answer))

    return entries

class RAD_Dataset(Dataset):
    def __init__(self, args,split):
        super(RAD_Dataset, self).__init__()

        self.args = args
        self.dataset = args.dataset

        config = yaml.load(open('data/config.yaml','r'),Loader=yaml.FullLoader)[args.dataset]
        
        ans2label_path = config['ans2label']
        label2ans_path = config['label2ans']
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))

        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = Dictionary.load_from_file(config['dictionary'])

        self.img2id = json.load(open(config['img2id'],'r'))

        self.entries = _load_dataset(config, split, self.img2id)
                
        # TODO: load images
        print('loading MAML image data ...')
        self.maml_images_data = cPickle.load(open(config['image84'], 'rb'))
        
        # TODO: load images
        print('loading DAE image data ...')
        self.ae_images_data = cPickle.load(open(config['image128'], 'rb'))

        # tokenization
        self.tokenize(args.question_len)
        self.tensorize()
#         if args.autoencoder and args.maml:
#             self.v_dim = args.v_dim * 2
#         else:
        print(args.v_dim)
        self.v_dim = args.v_dim * 2
        print(self.v_dim)

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        
        self.maml_images_data = torch.from_numpy(self.maml_images_data)
        self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')

        self.ae_images_data = torch.from_numpy(self.ae_images_data)
        self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        image_data = [0, 0]
        
        maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
        image_data[0] = maml_images_data

        ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
        image_data[1] = ae_images_data
    

        if answer_type == 'CLOSED':
            answer_target = 0
        else :
            answer_target = 1

        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return image_data,question, target, answer_type, question_type, answer_target

        else:
            return image_data, question, answer_type, question_type, answer_target

    def __len__(self):
        return len(self.entries)



class SEP_RAD_Dataset(Dataset):
    def __init__(self, args,split):
        super(SEP_RAD_Dataset, self).__init__()

        self.args = args
        self.dataset = args.dataset

        config = yaml.load(open('data/sep_config.yaml','r'),Loader=yaml.FullLoader)[args.dataset]
        
        open_ans2label_path = config['open_ans2label']
        open_label2ans_path = config['open_label2ans']

        close_ans2label_path = config['close_ans2label']
        close_label2ans_path = config['close_label2ans']

        self.open_ans2label = cPickle.load(open(open_ans2label_path, 'rb'))
        self.open_label2ans = cPickle.load(open(open_label2ans_path, 'rb'))
        self.close_ans2label = cPickle.load(open(close_ans2label_path, 'rb'))
        self.close_label2ans = cPickle.load(open(close_label2ans_path, 'rb'))

        self.num_open_ans_candidates = len(self.open_ans2label)
        self.num_close_ans_candidates = len(self.close_ans2label)

        self.dictionary = Dictionary.load_from_file(config['dictionary'])

        self.img2id = json.load(open(config['img2id'],'r'))

        self.entries = _load_dataset(config, split, self.img2id)
                
        # TODO: load images
        print('loading MAML image data ...')
        self.maml_images_data = cPickle.load(open(config['image84'], 'rb'))
        
        # TODO: load images
        print('loading DAE image data ...')
        self.ae_images_data = cPickle.load(open(config['image128'], 'rb'))

        # tokenization
        self.tokenize(args.question_len)
        self.tensorize()
#         if args.autoencoder and args.maml:
#             self.v_dim = args.v_dim * 2
#         else:
        self.v_dim = args.v_dim*2

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        
        self.maml_images_data = torch.from_numpy(self.maml_images_data)
        self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')

        self.ae_images_data = torch.from_numpy(self.ae_images_data)
        self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        image_data = [0, 0]
        
        maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
        image_data[0] = maml_images_data

        ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
        image_data[1] = ae_images_data
    

        if answer_type == 'CLOSED':
            answer_target = 0
        else :
            answer_target = 1

        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']

            if labels is not None:
                target = labels[0] 
            else:
                target = torch.tensor(1000000)
            
            # target = torch.zeros(self.num_ans_candidates)
            # if labels is not None:
            #     target.scatter_(0, labels, scores)

            return  image_data, question, target, answer_type, question_type, answer_target

        else:
            return image_data, question, answer_type, question_type, answer_target

    def __len__(self):
        return len(self.entries)

def get_dataset(args, dataset, split):
    if dataset == 'rad':
        if not args.tcr:
            return RAD_Dataset(args,split)
        else:
            return SEP_RAD_Dataset(args,split)




def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    if args.use_RAD:
        dataroot = args.RAD_dir
    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights






if __name__=='__main__':

    dataroot = './data_RAD'

    # d = Dictionary.load_from_file(os.path.join(dataroot,'dictionary.pkl'))
    # dataset = VQAFeatureDataset('train',d,dataroot)
    # train_data = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    # for i,row in enumerate(train_data):
    #     image_data,question, target, answer_type, question_type, phrase_type = row
    #     print(image_data.shape,question.shape,target.shape,answer_type,question_type,phrase_type)
    #     break
