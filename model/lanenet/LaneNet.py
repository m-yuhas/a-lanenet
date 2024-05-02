# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.loss import DiscriminativeLoss
from model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from model.lanenet.backbone.AnytimeENet import *
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(pytorch_lightning.LightningModule):
    def __init__(self, in_ch = 3, arch="ENet", base_model=""):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 3  # if you want to output RGB instance map, it should be 3.
        print("Use {} as backbone".format(arch))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)

            self._decoder_binary = Deeplabv3plus_Decoder(2)
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENetBlock2':
            m = LaneNet()
            m.load_state_dict(torch.load('log/base_model.pth'))
            self._encoder = ENet_EncoderBlock2(m._encoder)
            self._encoder.to(DEVICE)
            self._decoder_binary = ENet_DecoderEarlyExit(scale_factor=8, in_ch=128, out_ch=2)
            self._decoder_binary.to(DEVICE)

            self._decoder_instance = ENet_DecoderEarlyExit(scale_factor=8, in_ch=128, out_ch=self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENetBlock1':
            m = LaneNet()
            m.load_state_dict(torch.load('log/base_model.pth'))
            self._encoder = ENet_EncoderBlock1(m._encoder)
            self._encoder.to(DEVICE)
            self._decoder_binary = ENet_DecoderEarlyExit(scale_factor=4, in_ch=64, out_ch=2)

            self._decoder_instance = ENet_DecoderEarlyExit(scale_factor=4, in_ch=64, out_ch=self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENetBlock0':
            m = LaneNet()
            m.load_state_dict(torch.load('log/base_model.pth'))
            self._encoder = ENet_EncoderBlock0(m._encoder)
            self._encoder.to(DEVICE)
            self._decoder_binary = ENet_DecoderEarlyExit(scale_factor=2, in_ch=16, out_ch=2)
            self._decoder_binary.to(DEVICE)

            self._decoder_instance = ENet_DecoderEarlyExit(scaler_factor=2, in_ch=16, out_ch=self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        else:
            raise("Please select right model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch.startswith('ENet'):
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            # I think this is required
            #binary = self.sigmoid(binary)
            instance = self._decoder_instance(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        else:
            raise("Please select right model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }
