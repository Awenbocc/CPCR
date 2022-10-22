# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         language_model
# Description:  This code is from Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang's repository.
# https://github.com/jnhwkim/ban-vqa
# Author:       Boliu.Kelvin
# Date:         2020/4/7
#-------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import itertools
import json
import os
import yaml
from tools.create_dictionary import Dictionary
import itertools
import json
import os
import yaml

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, cat=True):
        super(WordEmbedding, self).__init__()
        self.cat = cat
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if cat:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if self.cat:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if self.cat:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb

class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        output, _ = self.rnn(x)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        output, _ = self.rnn(x)
        return output

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    # glove_file = glove_file if args.use_TDIUC else os.path.join(args.TDIUC_dir, 'glove', glove_file.split('/')[-1])
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def tfidf_loading(use_tfidf, w_emb, args):

    config = yaml.load(open('data/config.yaml','r'),Loader=yaml.FullLoader)[args.dataset]

    if use_tfidf:
        if args.use_data:
            dict = Dictionary.load_from_file(config['dictionary'])

        # load extracted tfidf and weights from file for saving loading time
        if args.use_data:
            if os.path.isfile(os.path.join(f'data/{args.dataset}', 'embed_tfidf_weights.pkl')) == True:
                print("Loading embedding tfidf and weights from file")
                with open(os.path.join(f'data/{args.dataset}', 'embed_tfidf_weights.pkl'), 'rb') as f:
                    w_emb = torch.load(f)
                print("Load embedding tfidf and weights from file successfully")
            else:
                print("Embedding tfidf and weights haven't been saving before")
                tfidf, weights = tfidf_from_questions(['train'], config, dict)
                w_emb.init_embedding(os.path.join(f'data/{args.dataset}', 'glove6b_init_300d.npy'), tfidf, weights)
                with open(os.path.join(f'data/{args.dataset}','embed_tfidf_weights.pkl'), 'wb') as f:
                    torch.save(w_emb, f)
                print("Saving embedding with tfidf and weights successfully")
    return w_emb

def tfidf_from_questions(names, config, dictionary, target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
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
            question_path = config[f'{name}_sample']
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
    glove_file = config['glove']
    weights, word2emb = create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights





