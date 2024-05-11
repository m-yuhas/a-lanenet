# Neeeded for Conversion
import argparse
import os
import torch
import torchvision
import pytorch_lightning
from typing import Callable, List, Tuple

from model.lanenet.LaneNet import LaneNet

class QuantizedInitialBlock(torch.nn.Module):
    def __init__(self, base_module):
        super(QuantizedInitialBlock, self).__init__()
        self.input_channel = base_module.input_channel
        self.conv_channel = base_module.conv_channel

        self.conv = base_module.conv
        self.maxpool = base_module.maxpool

        self.dq_maxp = torch.quantization.DeQuantStub()
        self.dq_conv = torch.quantization.DeQuantStub()
        self.q = torch.quantization.QuantStub()
    
    def forward(self, x):
        conv_branch = self.dq_conv(self.conv(x))
        maxp_branch = self.dq_maxp(self.maxpool(x))
        return self.q(torch.cat([conv_branch, maxp_branch], 1))


# TODO: add norm layers
class QuantizedBottleneck(torch.nn.Module):
    def __init__(self, base_module):
        super(QuantizedBottleneck, self).__init__()
        self.input_channel = base_module.input_channel
        self.activate = base_module.activate

        self.module_type = base_module.module_type
        if self.module_type == 'downsampling':
            self.maxpool = base_module.maxpool
            self.conv = base_module.conv
        elif self.module_type == 'upsampling':
            self.maxunpool = base_module.maxunpool
            self.conv = base_module.conv
        elif self.module_type == 'regular':
            self.conv = base_module.conv
        elif self.module_type == 'asymmetric':
            self.conv = base_module.conv
        elif self.module_type == 'dilated':
            self.conv = base_module.conv
        else:
            raise("Module Type error")
        self.dq_maxp = torch.quantization.DeQuantStub()
        self.dq_conv = torch.quantization.DeQuantStub()
        self.q_outp = torch.quantization.QuantStub()

    def forward(self, x):
        if self.module_type == 'downsampling':
            conv_branch = self.conv(x)
            maxp_branch = self.maxpool(x)
            bs, conv_ch, h, w = conv_branch.size()
            maxp_ch = maxp_branch.size()[1]
            padding = torch.zeros(bs, conv_ch - maxp_ch, h, w).to(maxp_branch.device)
            maxp_branch = self.dq_maxp(maxp_branch)
            padding= self.dq_maxp(padding)
            maxp_branch = torch.cat([maxp_branch, padding], 1)
            #maxp_branch = self.dq_maxp(maxp_branch)
            conv_branch = self.dq_conv(conv_branch)
            output = maxp_branch + conv_branch
            output = self.q_outp(output)
        elif self.module_type == 'upsampling':
            conv_branch = self.conv(x)
            maxunp_branch = self.maxunpool(x)
            maxunp_branch = self.dq_maxp(maxunp_branch)
            conv_branch = self.dq_conv(conv_branch)
            output = maxunp_branch + conv_branch
            output = self.q_outp(output)
        else:
            conv_branch = self.dq_conv(self.conv(x))
            x_branch = self.dq_maxp(x)
            output = conv_branch + x_branch
            output = self.q_outp(output)
        
        return self.activate(output)


class QuantizedENetDecoder(torch.nn.Module):
    
    def __init__(self, base_module):
        super(QuantizedENetDecoder, self).__init__()

        self.q0 = torch.quantization.QuantStub()
        self.bottleneck4_0 = QuantizedBottleneck(base_module.bottleneck4_0)
        self.bottleneck4_1 = QuantizedBottleneck(base_module.bottleneck4_1)
        self.bottleneck4_2 = QuantizedBottleneck(base_module.bottleneck4_2)

        self.bottleneck5_0 = QuantizedBottleneck(base_module.bottleneck5_0)
        self.bottleneck5_1 = QuantizedBottleneck(base_module.bottleneck5_1)

        self.upsample = base_module.upsample
        self.fullconv = base_module.fullconv
        self.dq0 = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.q0(x)
        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)

        x = self.upsample(x)
        x = self.fullconv(x)
        x = self.dq0(x)
        return x


class QuantizedENetDecoderEarlyExit(torch.nn.Module):
    
    def __init__(self, base_module):
        super(QuantizedENetDecoderEarlyExit, self).__init__()

        self.q0 = torch.quantization.QuantStub()
        self.pre_upsample = base_module.pre_upsample

        self.bottleneck5_0 = QuantizedBottleneck(base_module.bottleneck5_0)
        self.bottleneck5_1 = QuantizedBottleneck(base_module.bottleneck5_1)

        self.upsample = base_module.upsample
        self.fullconv = base_module.fullconv
        self.dq0 = torch.quantization.DeQuantStub()
    
    def forward(self, x):

        x = self.q0(x)
        x = self.pre_upsample(x)

        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)

        x = self.upsample(x)
        x = self.fullconv(x)
        x = self.dq0(x)
        return x


class LaneNetBlock0(torch.nn.Module):
    def __init__(self, full, b2, b1, b0):
        super().__init__()
        self.q = full._encoder.q0
        self.initial_block = QuantizedInitialBlock(full._encoder.initial_block)
        self.dq = full._encoder.dq0

        self.binary_decoder = QuantizedENetDecoderEarlyExit(b0._decoder_binary)
        self.instance_decoder = QuantizedENetDecoderEarlyExit(b0._decoder_instance)

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
        self.bottleneck1_0 = QuantizedBottleneck(full._encoder.bottleneck1_0)
        self.bottleneck1_1 = QuantizedBottleneck(full._encoder.bottleneck1_1)
        self.bottleneck1_2 = QuantizedBottleneck(full._encoder.bottleneck1_2)
        self.bottleneck1_3 = QuantizedBottleneck(full._encoder.bottleneck1_3)
        self.bottleneck1_4 = QuantizedBottleneck(full._encoder.bottleneck1_4)
        self.dq = full._encoder.dq1

        self.binary_decoder = QuantizedENetDecoderEarlyExit(b1._decoder_binary)
        self.instance_decoder = QuantizedENetDecoderEarlyExit(b1._decoder_instance)

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
        self.bottleneck2_0 = QuantizedBottleneck(full._encoder.bottleneck2_0)
        self.bottleneck2_1 = QuantizedBottleneck(full._encoder.bottleneck2_1)
        self.bottleneck2_2 = QuantizedBottleneck(full._encoder.bottleneck2_2)
        self.bottleneck2_3 = QuantizedBottleneck(full._encoder.bottleneck2_3)
        self.bottleneck2_4 = QuantizedBottleneck(full._encoder.bottleneck2_4)
        self.bottleneck2_5 = QuantizedBottleneck(full._encoder.bottleneck2_5)
        self.bottleneck2_6 = QuantizedBottleneck(full._encoder.bottleneck2_6)
        self.bottleneck2_7 = QuantizedBottleneck(full._encoder.bottleneck2_7)
        self.bottleneck2_8 = QuantizedBottleneck(full._encoder.bottleneck2_8)
        self.dq = full._encoder.dq2

        self.binary_decoder = QuantizedENetDecoderEarlyExit(b2._decoder_binary)
        self.instance_decoder = QuantizedENetDecoderEarlyExit(b2._decoder_instance)

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
        #self.q = full._encoder.q3
        self.q = torch.quantization.QuantStub()
        self.bottleneck3_0 = QuantizedBottleneck(full._encoder.bottleneck3_0)
        self.bottleneck3_1 = QuantizedBottleneck(full._encoder.bottleneck3_1)
        self.bottleneck3_2 = QuantizedBottleneck(full._encoder.bottleneck3_2)
        self.bottleneck3_3 = QuantizedBottleneck(full._encoder.bottleneck3_3)
        self.bottleneck3_4 = QuantizedBottleneck(full._encoder.bottleneck3_4)
        self.bottleneck3_5 = QuantizedBottleneck(full._encoder.bottleneck3_5)
        self.bottleneck3_6 = QuantizedBottleneck(full._encoder.bottleneck3_6)
        self.bottleneck3_7 = QuantizedBottleneck(full._encoder.bottleneck3_7)
        #self.dq = full._encoder.dq3
        self.dq = torch.quantization.DeQuantStub()

        self.binary_decoder = QuantizedENetDecoder(full._decoder_binary)
        self.instance_decoder = QuantizedENetDecoder(full._decoder_instance)

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
    for b in blocks:
        print(b)
    return blocks

def save_gpu_blocks(blocks):
    x = torch.randn(1,3,416,416).to(torch.device('cuda'))
    for idx, block in enumerate(blocks):
        #for layer in block.modules():
        #    if isinstance(layer, pytorch_lightning.LightningModule):
        #        layer._trainer = pytorch_lightning.Trainer()
        block = block.to(torch.device('cuda'))
        block = block.eval()
        b = torch.jit.trace(block, (x))
        torch.jit.save(b, f'lanenetblock{idx}.pt')
        if idx < 3:
            _, _, x = block(x)

def save_cpu_blocks(blocks):
    x = torch.randn(1,3,416,416).to(torch.device('cpu'))
    for idx, block in enumerate(blocks):
        #for layer in block.modules():
        #    if isinstance(layer, pytorch_lightning.LightningModule):
        #        layer._trainer = pytorch_lightning.Trainer()
        block = block.to(torch.device('cpu'))
        block = block.eval()
        b = torch.jit.trace(block, (x))
        torch.jit.save(b, f'lanenetblock{idx}q.pt')
        if idx < 3:
            _, _, x = block(x)

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
    save_gpu_blocks(blocks)
    #for idx, block in enumerate(blocks):
    #    #b = torch.jit.script(block)
    #    b = block.to_torchscript(method="trace")
    #    torch.save(b, f'lanenetblock{idx}.pt')
    #    #torch.save(block.state_dict(), f'lanenetblock{idx}.pt')
    print('Quantizing...')
    q_blocks = static_quantize(blocks)
    print('Saving quantized blocks...')
    save_cpu_blocks(q_blocks)
    #for idx, block in enumerate(blocks):
        #b = torch.jit.script(block)
        #torch.save(b, f'lanenetblock{idx}.pt')
        #torch.save(block.state_dict(), f'lanenetblock{idx}q.pt')
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save('model_scripted.pt') # Save