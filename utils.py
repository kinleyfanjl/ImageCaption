from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def save_models(encoder, middle, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file = os.path.join(checkpoint_path, 'model-%d-%d.ckpt' %(epoch+1, step+1))
    print('Saving model to:', checkpoint_file)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'middle_state_dict': middle.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer': optimizer,
        'step': step,
        'epoch': epoch,
        'losses_train': losses_train,
        'losses_val': losses_val
        }, checkpoint_file)

def load_models(checkpoint_file,sample=False):
    if sample:
        checkpoint = torch.load(checkpoint_file,map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_file)
    encoder_state_dict = checkpoint['encoder_state_dict']
    middle_state_dict = checkpoint['middle_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    optimizer = checkpoint['optimizer']
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    return encoder_state_dict, middle_state_dict, decoder_state_dict, optimizer, step, epoch, losses_train, losses_val

def dump_losses(losses_train, losses_val, path):
    import pickle
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'wb') as f:
        try:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f, protocol=2)
        except:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f)

def convert_back_to_text(idx_arr, vocab):
    '''
    from itertools import takewhile
    blacklist = [vocab.word2idx[word] for word in [vocab.start_token()]]
    predicate = lambda word_id: vocab.idx2word[word_id] != vocab.end_token()
    sampled_caption = [vocab.idx2word[word_id] for word_id in takewhile(predicate, idx_arr) if word_id not in blacklist]

    sentence = ' '.join(sampled_caption)
    '''
    sampled_caption = []
    for word_id in idx_arr:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence
