import random
import collections

import torch
import torch.nn as nn

# Parameters for the entire model (stem, all blocks, and head)
NetArgs = collections.namedtuple('NetArgs', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# Parameters for an individual model block
StageArgs = collections.namedtuple('StageArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# NAS or manually designed?
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t, r, sp):
        """
        Args:
            in_channels: input_channels
            sp = stochastic dropout ratio
            t = expansion factor
            r = se ratio (less than 1)
        """
        super().__init__()

        self.sp = sp
        # expansion phase (inverted residual block)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        # depthwise phase
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, kernel_size, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        # equeeze and excitation
        squeeze_channels = max(int(in_channels * t * r), 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels * t, squeeze_channels),
            Swish(),
            nn.Linear(squeeze_channels, in_channels * t),
            nn.Sigmoid()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        # expansion
        expansion = self.expansion(x)

        # depthwise
        depthwise = self.depthwise(expansion)

        # squeeze and excitation
        squeezed = self.squeeze(depthwise)
        squeezed = squeezed.view(squeezed.size(0), -1)
        excitation = self.excitation(squeezed)
        excitation = excitation.view(depthwise.size(0), depthwise.size(1), 1, 1)
        #print(excitation.shape)
        depthwise = depthwise * excitation

        #print(depthwise.shape)
        # pointwise
        pointwise = self.pointwise(depthwise)

        # stochastic depth
        if self.train:
            if random.random() < self.sp:
                x = shortcut + pointwise

            else:
                x = pointwise

        else:
            x = pointwise * self.sp + shortcut

        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes


net = MBConvBlock(3, 10, 5, 2, 6, 0.25, 0.5)

#swish = Swish()

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)

def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
      'efficientnet-b8': (2.2, 3.6, 672, 0.5),
      'efficientnet-l2': (4.3, 5.3, 800, 0.5),
  }
  return params_dict[model_name]

stage_args = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]
net_args = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}


img = torch.Tensor(4, 3, 20, 3)

print(net(img).shape)
