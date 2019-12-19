#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import random


class SSL_JIGSAW_COLORIZATION(object):
    def __init__(self, indices, permutation_nbr=1000):
        
        self.indices = set(indices)
        self.permutation_nbr = permutation_nbr
        self.permutations = self.__retrive_permutations(permutation_nbr)
        
        self.__augment_tile = transforms.Compose([
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        imgs, labels = [], []
        indices = self.indices if self.indices else set(range(len(sample["data"])))
        for idx in range(len(sample["data"])):
            if idx in indices:
                    label = torch.randint(self.permutation_nbr, [1]).item()
                    order = self.permutations[label]
                    img = sample["data"][idx]

                    s = float(img.size[0]) // 3
                    a = s // 2
                    tiles = [None] * 9
                    

                    for n in range(9):
                        i = n // 3 #row
                        j = n % 3 #col
                        # inputs are changed from 96*96*3 to 32*32*27 
                        c = [a * i * 2 + a, a * j * 2 + a]
                        c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
                        tile = img.crop(c.tolist())
                        if n not in order:
                            tile = self.__change_color(np.array(tile))
                        tile = self.__augment_tile(tile)
                        # Normalize the patches indipendently to avoid low level features shortcut
                        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
                        s[s == 0] = 1
                        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
                        tile = norm(tile)
                        tiles[n] = tile


                    for idx in range(9):
                        if order[idx] >= 9:
                            order[idx] -= 9

                    data = [tiles[order[t]] for t in range(9)]
                    data = torch.stack(data, 0)
                    
                    imgs.append(data)
                    labels.append(label)

        sample["data"] = imgs
        sample["label"] = labels

        return sample

    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_color_%d.npy' % (classes))

        return all_perm

    def __change_color(self, img):
        for ch in range(3):
            # color transformation1
            #img[:, :, ch] = img[:,:,ch] * np.random.uniform(1.5, 2.0)
            
            # color transformation2
            img[:, :, ch] = img[:, :, ch] * -1 * np.random.uniform() + 255
        img[img > 255] = 255
        return img.astype('uint8')


def create_permutation(permutation_nbr):
    permutation = np.load('permutations_1000.npy')
    
    for i in range(permutation_nbr):
        idx1, idx2 = random.sample(range(9), 2)

        permutation[i, idx1] += 9
        permutation[i, idx2] += 9
    
    np.save('permutation_color_%d.npy' % (permutation_nbr), permutation)

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

if __name__ == "__main__":
    create_permutation(1000)
