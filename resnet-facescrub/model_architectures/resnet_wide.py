"""
Wide ResNet for Split Learning on FaceScrub 64x64.

Based on CifarResNet but with 4x wider channels:
  Original: 16 -> 32 -> 64 -> 128   (~1.7M params for depth=110)
  Wide(4x): 64 -> 128 -> 256 -> 512  (~27M params for depth=110)

WideResNet28_10 (depth=28, widen=10) is the classic "best ResNet" from:
  "Wide Residual Networks" (Zagoruyko & Komodakis, 2016)

For 64x64 FaceScrub input:
  conv7x7(stride=2) -> 32x32
  stage_1(stride=2) -> 16x16
  stage_2(stride=2) -> 8x8
  stage_3(stride=2) -> 4x4
  adaptive_avg_pool -> 1x1

With cutlayer=4 + C8S1 bottleneck:
  smashed data = [B, 8, 16, 16] = 2048 dim (same as VGG-11 on FaceScrub)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import copy
import numpy as np
from thop import profile


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class DownsampleConv(nn.Module):
    """Downsample with 1x1 conv (proper residual, not zero-padding)."""
    def __init__(self, nIn, nOut, stride):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + out, inplace=True)


class WideResNet(nn.Module):
    """
    Wide ResNet for split learning.

    Args:
        depth: total depth (must satisfy (depth-4)%6==0)
        widen_factor: channel width multiplier
        cutting_layer: split point between local and cloud
        num_class: number of output classes
        dropout: dropout rate inside blocks
    """

    def __init__(self, depth, widen_factor, cutting_layer, logger,
                 num_client=1, num_class=10, initialize_different=False,
                 adds_bottleneck=False, bottleneck_option="C8S1",
                 sgm=True, double_local_layer=False, upsize=False,
                 SCA=False, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, f'WideResNet depth must satisfy (d-4)%6==0, got {depth}'
        n_blocks = (depth - 4) // 6  # blocks per stage

        # Channel widths: base [16, 32, 64, 128] * widen_factor for stages
        nStages = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]

        print(f'WideResNet-{depth}-{widen_factor}: blocks/stage={n_blocks}, '
              f'channels={nStages}, dropout={dropout}')

        self.current_client = 0
        self.inplanes = nStages[0]

        layers = []
        # Initial conv: 7x7 stride=2 for 64x64 input (reduces to 32x32)
        layers.append(nn.Sequential(
            nn.Conv2d(3, nStages[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(nStages[0]),
            nn.ReLU(inplace=True)
        ))

        if double_local_layer:
            layers.append(nn.Conv2d(nStages[0], nStages[0], 3, padding=1, bias=False))
            cutting_layer += 1

        # Three stages
        stage_1 = self._make_layer(BasicBlock, nStages[1], n_blocks, stride=2, dropout=dropout)
        layers.extend(stage_1)
        stage_2 = self._make_layer(BasicBlock, nStages[2], n_blocks, stride=2, dropout=dropout)
        layers.extend(stage_2)
        stage_3 = self._make_layer(BasicBlock, nStages[3], n_blocks, stride=2, dropout=dropout)
        layers.extend(stage_3)

        # Split into local / cloud
        try:
            local_layer_list = layers[:cutting_layer]
            cloud_layer_list = layers[cutting_layer:]
        except:
            print("Cutting layer exceeds model depth, setting cloud to empty")
            local_layer_list = layers[:]
            cloud_layer_list = []

        # Probe for in_channels at split point
        temp_local = nn.Sequential(*local_layer_list)
        with torch.no_grad():
            probe = torch.randn([1, 3, 64, 64])  # FaceScrub 64x64
            smashed = temp_local(probe)
        in_channels = smashed.size(1)
        print(f"in_channels at cut point: {in_channels}")

        # Bottleneck
        local = []
        cloud = []
        if adds_bottleneck:
            print(f"original channel size of smashed-data is {in_channels}")
            try:
                relu_option = "noRELU" not in bottleneck_option and "norelu" not in bottleneck_option
                if "K" in bottleneck_option:
                    bn_kernel_size = int(bottleneck_option.split("C")[0].split("K")[1])
                else:
                    bn_kernel_size = 3
                bottleneck_channel_size = int(bottleneck_option.split("S")[0].split("C")[1])
                bottleneck_stride = int(bottleneck_option.split("S")[1]) if "S" in bottleneck_option else 1
            except:
                print("auto extract bottleneck option fail, set to 8")
                bn_kernel_size = 3
                bottleneck_channel_size = 8
                bottleneck_stride = 1
                relu_option = True

            if bottleneck_stride == 1:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size,
                                    kernel_size=bn_kernel_size, padding=bn_kernel_size // 2, stride=1)]
            elif bottleneck_stride >= 2:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=3, padding=1, stride=2)]
                for _ in range(int(np.log2(bottleneck_stride // 2))):
                    if relu_option:
                        local += [nn.ReLU()]
                    local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride=2)]
            if relu_option:
                local += [nn.ReLU()]

            if bottleneck_stride == 1:
                cloud += [nn.Conv2d(bottleneck_channel_size, in_channels,
                                    kernel_size=bn_kernel_size, padding=bn_kernel_size // 2, stride=1)]
            elif bottleneck_stride >= 2:
                for _ in range(int(np.log2(bottleneck_stride // 2))):
                    cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size,
                                                 kernel_size=3, output_padding=1, padding=1, stride=2)]
                    if relu_option:
                        cloud += [nn.ReLU()]
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, in_channels,
                                             kernel_size=3, output_padding=1, padding=1, stride=2)]
            if relu_option:
                cloud += [nn.ReLU()]
            print(f"added bottleneck, new channel size of smashed-data is {bottleneck_channel_size}")

        if sgm:
            local += [nn.Sigmoid()]

        local_layer_list += local
        cloud_layer_list = cloud + cloud_layer_list

        self.local = nn.Sequential(*local_layer_list)
        self.cloud = nn.Sequential(*cloud_layer_list)

        # Multi-client support
        self.local_list = []
        for i in range(num_client):
            if i == 0:
                self.local_list.append(self.local)
                self.local_list[0].apply(_init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])
                self.local_list.append(new_copy.cuda())
                if initialize_different:
                    self.local_list[i].apply(_init_weights)

        self.logger = logger
        self.classifier = nn.Linear(nStages[3] * BasicBlock.expansion, num_class)

        print("local:")
        print(self.local)
        print("cloud:")
        print(self.cloud)
        print("classifier:")
        print(self.classifier)

        # Init cloud weights
        for m in self.cloud:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleConv(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))
        return layers

    def switch_model(self, client_id):
        self.current_client = client_id
        self.local = self.local_list[client_id]

    def get_smashed_data_size(self):
        with torch.no_grad():
            device = next(self.local.parameters()).device
            try:
                noise_input = torch.randn([1, 3, 64, 64]).to(device)
                smashed_data = self.local(noise_input)
            except:
                noise_input = torch.randn([1, 3, 32, 32]).to(device)
                smashed_data = self.local(noise_input)
        return smashed_data.size()

    def get_MAC_param(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 64, 64])
            device = next(self.local.parameters()).device
            noise_input = noise_input.to(device)
            client_macs, client_params = profile(self.local, inputs=(noise_input,))
            noise_smash = torch.randn(self.get_smashed_data_size())
            device = next(self.cloud.parameters()).device
            noise_smash = noise_smash.to(device)
            server_macs, server_params = profile(self.cloud, inputs=(noise_smash,))
            noise_final = self.cloud(noise_smash)
            noise_final = F.adaptive_avg_pool2d(noise_final, 1)
            noise_final = noise_final.view(noise_final.size(0), -1)
            clas_macs, clas_params = profile(self.classifier, inputs=(noise_final,))
            server_macs += clas_macs
            server_params += clas_params
        return client_macs, client_params, server_macs, server_params

    def forward(self, x):
        self.local_output = self.local(x)
        x = self.cloud(self.local_output)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ── Factory functions ────────────────────────────────────────────────────────

def WideResNet28_10(cutting_layer, logger, num_client=1, num_class=10,
                    initialize_different=False, adds_bottleneck=False,
                    bottleneck_option="C8S1", double_local_layer=False,
                    upsize=False, SCA=False):
    """WideResNet-28-10: depth=28, widen=10. ~36M params. Classic best WRN."""
    return WideResNet(28, 10, cutting_layer, logger,
                      num_client=num_client, num_class=num_class,
                      initialize_different=initialize_different,
                      adds_bottleneck=adds_bottleneck,
                      bottleneck_option=bottleneck_option,
                      double_local_layer=double_local_layer,
                      upsize=upsize, SCA=SCA, dropout=0.3)


def WideResNet16_8(cutting_layer, logger, num_client=1, num_class=10,
                   initialize_different=False, adds_bottleneck=False,
                   bottleneck_option="C8S1", double_local_layer=False,
                   upsize=False, SCA=False):
    """WideResNet-16-8: depth=16, widen=8. ~11M params. Lighter alternative."""
    return WideResNet(16, 8, cutting_layer, logger,
                      num_client=num_client, num_class=num_class,
                      initialize_different=initialize_different,
                      adds_bottleneck=adds_bottleneck,
                      bottleneck_option=bottleneck_option,
                      double_local_layer=double_local_layer,
                      upsize=upsize, SCA=SCA, dropout=0.3)
