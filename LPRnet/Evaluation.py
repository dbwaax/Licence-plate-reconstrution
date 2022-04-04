#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""

from LPRnet.model.LPRNET import LPRNet, CHARS
from LPRnet.model.STN import STNet
from LPRnet.load_data import LPRDataLoader, collate_fn
# from load_data_copy import LPRDataLoader, collate_fn
# from data.load_data import LPRDataLoader, collate_fn
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:,:,::-1]
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataLoader([args.img_dirs], args.img_size)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        imgs, labels, lengths = next(iter(dataloader))
        
        input_tensor = imgs.cpu()
        transformed_input_tensor = STN(imgs.to(device)).cpu()
        
        in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        
        # Plot the results side-by-side
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
        
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels
def decodess(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        pred_labels.append(pred_label)
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels
def eval(lprnet, STN,generate,dataloader, lens, device): 
    lprnet = lprnet.to(device)
    STN = STN.to(device)
    TP = 0
    for imgs,_,labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        with torch.no_grad():
            imgs, labels = imgs.to(device), labels.to(device)
            fake_B = generate(imgs)
            transfer = STN(fake_B)
            logits = lprnet(transfer) # torch.Size([batch_size, CHARS length, output length ])
        
            preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
            _, pred_labels = decode(preds, CHARS)  # list of predict output

            start = 0
            for i, length in enumerate(lengths):
                label = labels[start:start+length]
                start += length
                if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                    TP += 1
    ACC = TP / lens
    
    return ACC

