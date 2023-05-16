#!/usr/bin/python
# encoding: utf-8


import collections
from PIL import Image
import numpy as np



class resizeNormalize(object):
    
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):

        size = self.size
        imgW, imgH = size
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        if w <= imgW:
            newImage = np.zeros((imgH, imgW), dtype='uint8')
            newImage[:] = 255
            newImage[:, :w] = np.array(img)
            img = Image.fromarray(newImage)
        else:
            img = img.resize((imgW, imgH), self.interpolation)

        img = np.array(img,dtype=np.float32)

        img -= 127.5
        img /= 127.5


        img = img.reshape([*img.shape,1])

        return img


class strLabelConverter(object):
    
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    
    def decode(self, t, length, raw=False, char_pix=4):
        t = t[:length]
        raw = ''.join([self.alphabet[i - 1] for i in t])
        char_list = []
        for i in range(length):

            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char = self.alphabet[t[i] - 1]
                start = i * char_pix
                char_list.append({'char': char, 'start': start, 'idx': i})
        
        line_pred = ''.join(dic['char'] for dic in char_list)
        return {'raw': raw, 'line_pred': line_pred, 'char_list': char_list}





class averager(object):
    
    def __init__(self):
        self.reset()
    
    def add(self, v):
        self.n_count += v.data.numel()
        # NOTE: not `+= v.sum()`, which will add a node in the compute graph,
        # which lead to memory leak
        self.sum += v.data.sum()
    
    def reset(self):
        self.n_count = 0
        self.sum = 0
    
    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

