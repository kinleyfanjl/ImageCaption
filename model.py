import torch
from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class CNN(nn.Module):
    """Class to build new model including all but last layers"""
    def __init__(self, output_dim = 1024):
        # initilization of base classes
        super(CNN, self).__init__()
        # use ResNet34 as the default CNN part
        # CNN transforms images to a certain feature
        pretrained_model = models.resnet34(pretrained=True)
        # ResNet34 component
        self.resnet = Sequential(*list(pretrained_model.children())[:-1]) 
        # nn.Linear: linear transform, with two parameters inputSize & outputSize
        self.linear = nn.Linear(pretrained_model.fc.in_features, output_dim)

        # batch normalization layer
        self.batchnorm = nn.BatchNorm1d(pretrained_model.fc.in_features, momentum=0.01)
        self.output = pretrained_model.fc.in_features

        self.init_weights()

    def init_weights(self):
        # weight initialization
        self.linear.weight.data.normal_(0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.resnet(x)
        x = Variable(x.data)
        x = x.view(x.size(0), -1) 
        x = self.linear(x)
        return x
    
class fcNet(nn.Module):
    """
    Fully connect layer for feature merging.
    """
    def __init__(self, n_feature, n_hidden1, n_output):
        super(fcNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.bh = torch.nn.Batchnorm1d(n_hidden1)
        self.output = torch.nn.Linear(n_hidden1, n_output)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden1(x))
        x = self.output(torch.nn.functional.relu(self.bh(x)))
        return(x)

class RNN(nn.Module):
    """
    Recurrent Neural Network for Text Generation.
    As an Encoder-Decoder Network for Image Captioning.
    """
    __rec_units = { 'gru': nn.GRU, 'lstm': nn.LSTM }

    def __init__(self, emb_size, hidden_size, vocab_size, num_layers=1, rec_unit='gru', drop_out = 0.1):
        """
        Initializer
        :param emb_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param num_layers: number of recurrent layers (default=1)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        rec_unit = rec_unit.lower()
        assert rec_unit in RNN.__rec_units, 'Specified recurrent unit is not available'

        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.unit = RNN.__rec_units[rec_unit](emb_size, hidden_size, num_layers,
                                                 batch_first=True, dropout = drop_out)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        Forward pass through the network
        :param features: features from CNN feature extractor
        :param captions: encoded and padded (target) image captions
        :param lengths: actual lengths of image captions
        :return: predicted distributions over the vocabulary
        """
        # embedding tokens in vector space
        embeddings = self.embeddings(captions)
        # append image as first input
        # unsqueeze: increase dimension by 1
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        # pack data (prepare it for pytorch model)
        inputs_packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        # run data through recurrent network
        hiddens, _ = self.unit(inputs_packed)
        outputs = self.linear(hiddens[0])
        return outputs
    def sample(self, features, max_len=25):
        """
        Sample from Recurrent network using greedy decoding
        :param features: features from CNN feature extractor
        :returns: predicted image captions
        """
        output_ids = []
        states = None
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.unit(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # find maximal predictions
            predicted = outputs.max(1)[1]
            # append results from given step to global results
            output_ids.append(predicted)
            # prepare chosen words for next decoding step
            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1)
            
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze()