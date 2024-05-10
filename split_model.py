# Neeeded for Conversion
import argparse
import os
import torch
import torchvision
import pytorch_lightning
from typing import Callable, List, Tuple

from model.lanenet.LaneNet import LaneNet

# TODO: add norm layers

class LaneNetBlock0(torch.nn.Module):
    def __init__(self, full, b2, b1, b0):
        super().__init__()
        self.q = full._encoder.q0
        self.initial_block = full._encoder.initial_block
        self.dq = full._encoder.dq0

        self.binary_decoder = b0._decoder_binary
        self.instance_decoder = b0._decoder_instance

    def forward(self, x):
        x0 = self.q(x)
        x0 = self.initial_block(x0)
        x0 = self.dq(x0)

        bin_out = self.binary_decoder(x0)
        inst_out = self.instance_decoder(x0)

        return bin_out, inst_out, x0


class LaneNetBlock1(torch.nn.Module):
    def __init__(self, full, b2, b1, b0):
        super().__init__()
        self.q = full._encoder.q1
        self.bottleneck1_0 = full._encoder.bottleneck1_0
        self.bottleneck1_1 = full._encoder.bottleneck1_1
        self.bottleneck1_2 = full._encoder.bottleneck1_2
        self.bottleneck1_3 = full._encoder.bottleneck1_3
        self.bottleneck1_4 = full._encoder.bottleneck1_4
        self.dq = full._encoder.dq1

        self.binary_decoder = b1._decoder_binary
        self.instance_decoder = b1._decoder_instance

    def forward(self, x0):
        x1 = self.q(x0)
        x1 = self.bottleneck1_0(x1)
        x1 = self.bottleneck1_1(x1)
        x1 = self.bottleneck1_2(x1)
        x1 = self.bottleneck1_3(x1)
        x1 = self.bottleneck1_4(x1)
        x1 = self.dq(x1)

        bin_out = self.binary_decoder(x1)
        inst_out = self.instance_decoder(x1)

        return bin_out, inst_out, x1


class LaneNetBlock2(torch.nn.Module):
    def __init__(self, full, b2, b1, b0):
        super().__init__()
        self.q = full._encoder.q2
        self.bottleneck2_0 = full._encoder.bottleneck2_0
        self.bottleneck2_1 = full._encoder.bottleneck2_1
        self.bottleneck2_2 = full._encoder.bottleneck2_2
        self.bottleneck2_3 = full._encoder.bottleneck2_3
        self.bottleneck2_4 = full._encoder.bottleneck2_4
        self.bottleneck2_5 = full._encoder.bottleneck2_5
        self.bottleneck2_6 = full._encoder.bottleneck2_6
        self.bottleneck2_7 = full._encoder.bottleneck2_7
        self.bottleneck2_8 = full._encoder.bottleneck2_8
        self.dq = full._encoder.dq2

        self.binary_decoder = b2._decoder_binary
        self.instance_decoder = b2._decoder_instance

    def forward(self, x1):
        x2 = self.q(x1)
        x2 = self.bottleneck2_0(x2)
        x2 = self.bottleneck2_1(x2)
        x2 = self.bottleneck2_2(x2)
        x2 = self.bottleneck2_3(x2)
        x2 = self.bottleneck2_4(x2)
        x2 = self.bottleneck2_5(x2)
        x2 = self.bottleneck2_6(x2)
        x2 = self.bottleneck2_7(x2)
        x2 = self.bottleneck2_8(x2)
        x2 = self.dq(x2)

        bin_out = self.binary_decoder(x2)
        inst_out = self.instance_decoder(x2)

        return bin_out, inst_out, x2

    
class LaneNetBlock3(torch.nn.Module):
    def __init__(self, full, b2, b1, b0):
        super().__init__()
        self.q = full._encoder.q3
        self.bottleneck3_0 = full._encoder.bottleneck3_0
        self.bottleneck3_1 = full._encoder.bottleneck3_1
        self.bottleneck3_2 = full._encoder.bottleneck3_2
        self.bottleneck3_3 = full._encoder.bottleneck3_3
        self.bottleneck3_4 = full._encoder.bottleneck3_4
        self.bottleneck3_5 = full._encoder.bottleneck3_5
        self.bottleneck3_6 = full._encoder.bottleneck3_6
        self.bottleneck3_7 = full._encoder.bottleneck3_7
        self.dq = full._encoder.dq3

        self.binary_decoder = full._decoder_binary
        self.instance_decoder = full._decoder_instance

    def forward(self, x2):
        x3 = self.q(x2)
        x3 = self.bottleneck3_0(x3)
        x3 = self.bottleneck3_1(x3)
        x3 = self.bottleneck3_2(x3)
        x3 = self.bottleneck3_3(x3)
        x3 = self.bottleneck3_4(x3)
        x3 = self.bottleneck3_5(x3)
        x3 = self.bottleneck3_6(x3)
        x3 = self.bottleneck3_7(x3)
        x3 = self.dq(x3)

        bin_out = self.binary_decoder(x3)
        inst_out = self.instance_decoder(x3)

        return bin_out, inst_out


def split_model(full: str, b2: str, b1: str, b0: str) -> List[torch.nn.Module]:
    sd_full = torch.load(full)
    sd_b2 = torch.load(b2)
    sd_b1 = torch.load(b1)
    sd_b0 = torch.load(b0)
    full = LaneNet()
    b2 = LaneNet(arch='ENetBlock2')
    b1 = LaneNet(arch='ENetBlock1')
    b0 = LaneNet(arch='ENetBlock0')
    full.load_state_dict(sd_full)
    b2.load_state_dict(sd_b2)
    b1.load_state_dict(sd_b1)
    b0.load_state_dict(sd_b0)
    return [
        LaneNetBlock0(full, b2, b1, b0),
        LaneNetBlock1(full, b2, b1, b0),
        LaneNetBlock2(full, b2, b1, b0),
        LaneNetBlock3(full, b2, b1, b0),
    ]


def static_quantize(blocks: Tuple[torch.nn.Module]) -> Tuple[torch.nn.Module]:
    blocks = [b.to('cpu').eval() for b in blocks]
    #for name, param in blocks[3].named_parameters():
    #    print(name)

    # Fuse Conv, bn and relu
    #blocks = [torch.quantization.fuse_modules(b, fuse_list) for b, fuse_list in zip(blocks, fuse_lists)]

    # Specify quantization configuration
    for block in blocks:
        block.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        #block.qconfig = torch.quantization.get_default_qconfig('x86')

    blocks = [torch.quantization.prepare(b) for b in blocks]

    # Calibrate with the training set
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((416, 416)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    folder_names = [
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town03',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town03_Opt',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town04',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town04_Opt',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town05',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town05_Opt',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town06',
        '/mnt/sdb/Datasets/CARLANE/MuLane/data/val/source/Town06_Opt',
    ]
    for f in folder_names:
        cal_set = torchvision.datasets.ImageFolder(
            f,
            transform=transforms,
            is_valid_file=lambda x: True if x.endswith('.jpg') else False
        )
        data_loader = torch.utils.data.DataLoader(
            cal_set,
            batch_size=16,
            shuffle=True
        )
        with torch.no_grad():
            count = 0
            for x, _ in data_loader:
                print(f'Calibration Batch: {count}')
                count += 1
                _, _, x0, = blocks[0](x)
                _, _, x1 = blocks[1](x0)
                _, _, x2 = blocks[2](x1)
                _, _ = blocks[3](x2)

    # Convert to quantized model
    blocks = [torch.quantization.convert(b) for b in blocks]
    return blocks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantize a Beta-VAE model')
    parser.add_argument(
        '--full',
        help='Weights for full model',
    )
    parser.add_argument(
        '--b2',
        help='Weights for block2'
    )
    parser.add_argument(
        '--b1',
        help='Weights for block1'
    )
    parser.add_argument(
        '--b0',
        help='Weights for block0'
    )
    # TODO: Allow selection of calibration set
    #parser.add_argument(
    #    '--calset',
    #    help='Calibration set for static quantization'
    #)
    args = parser.parse_args()
    print('Splitting model...')
    blocks = split_model(
        args.full,
        args.b2,
        args.b1,
        args.b0,
    )
    print('Saving blocks...')
    for idx, block in enumerate(blocks):
        torch.save(block, f'lanenetblock{idx}.pt')
    print('Quantizing...')
    q_blocks = static_quantize(blocks)
    print('Saving quantized blocks...')
    for idx, block in enumerate(blocks):
        torch.save(block, f'lanenetblock{idx}q.pt')
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save('model_scripted.pt') # Save