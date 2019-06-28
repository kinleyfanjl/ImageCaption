import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import re
from PIL import Image
from vocab import Vocabulary
import matplotlib.pyplot as plt
import json


def merge(image1, image2):
    w, h = image1.size[0], image1.size[1]
    merge_img = Image.new('RGB', (w * 2, h), 0xffffff)
    merge_img.paste(image1, (0, 0))
    merge_img.paste(image2, (w, 0))
    return merge_img

class ImageDataset(data.Dataset):
    """Dataset of Images"""
    def __init__(self, path, mode, vocab=None, transform=None):
        """
        :param path: images directory path
        :param json: captions file path
        :param vocab: vocabulary
        :param transform: a torchvision.transforms image transformer for preprocessing
        """
        self.path = path
        self.vocab = vocab
        self.transform = transform
        self.mode = mode
        json_file = "relative_captions_shoes.json"
        with open(json_file, 'r') as f:
            print("Load json file from {}".format(json_file))
            data = json.load(f)
        ###
        target =  [ind['ImageName'] for ind in data]
        refer =  [ind['ReferenceImageName'] for ind in data]
        caption = [ind['RelativeCaption'] for ind in data]
        total_len = len(caption)
        if mode == 'train':
            idx = slice(0, int(0.8 * total_len))
            #idx = slice(0, 100)
        elif mode == "valid":
            idx = slice(int(0.8 * total_len), int(0.9 * total_len))
            #idx = slice(100,150)
        else:
            idx = slice(int(0.9 * total_len), total_len)
            #idx = slice(150,200)

        ###
        pattern = re.compile('_\d+')
        ### image path
        target_path = ['attributedata/' + ind[4:( pattern.search(ind).start())] + '/' + str((int(pattern.search(ind).group(0)[1:]) > 999)+0) + '/' + ind
                      for ind in target]
        refer_path = ['attributedata/' + ind[4:( pattern.search(ind).start())] + '/' + str((int(pattern.search(ind).group(0)[1:]) > 999)+0) + '/' + ind
                      for ind in refer]
        self.target_path = target_path[idx]
        self.refer_path = refer_path[idx]
        self.caption = caption[idx]

    def __getitem__(self,index):
        """special python method for indexing a dict. dict[index]
        :param index: dataset id
        return: (image, caption)
        """
        target_path = self.target_path
        refer_path = self.refer_path
        vocab = self.vocab
        caption = self.caption
        '''
        here take the differences of target and refer_images
        '''
        target_image = Image.open(target_path[index])
        refer_image = Image.open(refer_path[index])
        '''
        ### another choice
        image1 = merge(target_image, refer_image)
        image2 = merge(refer_image, target_image)
        image = np.array(image1) - np.array(image2)
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        '''
        image = merge(refer_image, target_image)
        caption_ind = caption[index]

        if self.transform != None:
            # apply image preprocessing
            image = self.transform(image)
        # tokenize captions
        caption_str = str(caption_ind).lower()
        tokens = nltk.tokenize.word_tokenize(caption_str)
        caption_voc = torch.Tensor([vocab(vocab.start_token())] +
                               [vocab(token) for token in tokens] +
                               [vocab(vocab.end_token())])

        return image, caption_voc

    def __len__(self):
        return len(self.caption)

def collate_fn(data):
    """Create mini-batches of (target_image, refer_image, caption)
    Custom collate_fn for torch.utils.data.DataLoader is necessary for padding captions
    :param data: list; (target_image, refer_image, caption) tuples
            - target_image: tensor;    3 x 280 x 280 
            - refer_image: tensor;    3 x 280 x 280 
            - caption: tensor;  1 x length_caption
    Return: mini-batch
    :return target_images: tensor;     batch_size x 3 x 280 x 280
    :return refer_images: tensor;     batch_size x 3 x 280 x 280
    :return padded_captions: tensor;    batch_size x length
    :return caption_lengths: list;      lenghths of actual captions (without padding)
    """

    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge image tensors (stack)
    images = torch.stack(images, 0)

    # Merge captions
    caption_lengths = [len(caption) for caption in captions]

    # zero-matrix num_captions x caption_max_length
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()

    # fill the zero-matrix with captions. the remaining zeros are padding
    for idx, caption in enumerate(captions):
        end = caption_lengths[idx]
        padded_captions[idx, :end] = caption[:end]
    return images, padded_captions, caption_lengths
    
def data_and_loader(path, mode, vocab, transform = None,
        batch_size = 32, shuffle = True, num_workers = 0):
    """Returns Image Dataloader"""

    data = ImageDataset(path=path,
                        mode=mode,
                        vocab=vocab,
                        transform=transform,
                       )

    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data, data_loader
