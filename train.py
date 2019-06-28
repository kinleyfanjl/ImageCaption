from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn as nn
import numpy as np
import utils
from model import CNN, RNN, fcNet
from vocab import Vocabulary, build_vocab
from data import data_and_loader
import os
import json

def main(batch_size, embed_size, num_hiddens, num_layers, ln_hidden, ln_output,
         rec_unit, learning_rate = 1e-4, log_step = 10,
         num_epochs = 50, save_step = 100, ngpu = 1):
    # hyperparameters
    num_workers = 0
    checkpoint_dir = 'checkpoint'
    # Image Preprocessing

    transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    }
    # load data
    vocab = build_vocab(path='relative_captions_shoes.json')
    train_data, train_loader = data_and_loader(path='relative_captions_shoes.json', 
                                               mode = 'train', 
                                               vocab = vocab, 
                                               transform = transform['train'],
                                               batch_size = batch_size)

    val_data, val_loader = data_and_loader(path='relative_captions_shoes.json', 
                                           mode = 'valid', 
                                           vocab = vocab, 
                                           transform = transform['val'],
                                           batch_size = batch_size)


    losses_val = []
    losses_train = []

    # Build the models
    initial_step = initial_epoch = 0

    encoder = CNN(embed_size) ### embed_size: power of 2
    middle = fcNet(embed_size, ln_hidden, ln_output)
    decoder = RNN(ln_output, num_hiddens, len(vocab), num_layers, rec_unit=rec_unit, drop_out = 0.1)

    # Loss, parameters & optimizer
    loss_fun = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batchnorm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    
    # Train the Models
    total_step = len(train_loader)
    try:
        for epoch in range(initial_epoch, num_epochs):
            print('Epoch: {}'.format(epoch))
            for step, (images, captions, lengths) in enumerate(train_loader, start=initial_step):

                # Set mini-batch dataset
                images = Variable(images)      
                captions = Variable(captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                # Forward, Backward and Optimize
                decoder.zero_grad()
                middle.zero_grad()
                encoder.zero_grad()

                if ngpu > 1:
                    # run on multiple GPUs
                    features = nn.parallel.data_parallel(encoder, images, range(ngpu))                  
                    rnn_input = nn.parallel.data_parallel(middle, features, range(ngpu))
                    outputs = nn.parallel.data_parallel(decoder, features, range(ngpu))
                else:
                    # run on single GPU
                    features = encoder(images)
                    rnn_input = middle(features)
                    outputs = decoder(rnn_input, captions, lengths)

                train_loss = loss_fun(outputs, targets)
                losses_train.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

                # Run validation set and predict
                if step % log_step == 0:
                    encoder.batchnorm.eval()
                    # run validation set
                    batch_loss_val = []
                    for val_step, (images, captions, lengths) in enumerate(val_loader):
                        images = Variable(images)      
                        captions = Variable(captions)
                        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        #features = encoder(target_images) - encoder(refer_images)
                        features = encoder(images)
                        rnn_input = middle(features)
                        outputs = decoder(rnn_input, captions, lengths)
                        val_loss = loss_fun(outputs, targets)
                        batch_loss_val.append(val_loss.item())

                    losses_val.append(np.mean(batch_loss_val))

                    # predict
                    sampled_ids = decoder.sample(rnn_input)
                    sampled_ids = sampled_ids.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(sampled_ids, vocab)
                    print('Sample:', sentence)

                    true_ids = captions.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(true_ids, vocab)
                    print('Target:', sentence)

                    print('Epoch: {} - Step: {} - Train Loss: {} - Eval Loss: {}'.format(epoch, step, losses_train[-1], losses_val[-1]))
                    encoder.batchnorm.train()

                # Save the models
                if (step+1) % save_step == 0:
                    save_models(encoder, middle, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_dir)
                    dump_losses(losses_train, losses_val, os.path.join(checkpoint_dir, 'losses.pkl'))

    except KeyboardInterrupt:
        pass
    finally:
        # Do final save
        utils.save_models(encoder, middle, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_dir)
        utils.dump_losses(losses_train, losses_val, os.path.join(checkpoint_dir, 'losses.pkl'))

if __name__ == '__main__':
    main(batch_size=5, embed_size=1024, num_hiddens=512, num_layers = 2, ln_hidden=1024, ln_output=512, log_step=1, rec_unit='gru',
         learning_rate=1e-4, num_epochs=50, save_step=1000, ngpu=1)
