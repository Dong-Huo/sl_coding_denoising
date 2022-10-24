import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import torch.nn.functional as F
from models.modules import *


class disp_net(nn.Module):

    def __init__(self, enc_chs=[10, 32, 64, 128, 256], dec_chs=[256, 128, 64, 32], out_c=1, top_k=10):
        super(disp_net, self).__init__()
        enc_chs[0] = top_k
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.tail = nn.Conv2d(dec_chs[-1], out_c, 1)
        self.down_scales = 2 ** len(dec_chs)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape

        hb, wb = self.down_scales, self.down_scales
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.tail(out)
        # out = self.crop(out, x)
        return out[:, :, :h_inp, :w_inp]

