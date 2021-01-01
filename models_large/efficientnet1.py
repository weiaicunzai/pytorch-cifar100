import re
import math
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
#class MBConvBlock(torch.jit.ScriptModule):
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t, r, sp, m, eps):
        """
        Args:
            in_channels: input_channels
            sp = stochastic dropout ratio
            t = expansion factor
            r = se ratio (less than 1)
            m = batch norm momentum
            eps = batch norm epsilon
        """
        super().__init__()

        self.sp = 1 - sp
        # expansion phase (inverted residual block)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t, momentum=m, eps=eps),
            Swish()
        )

        # depthwise phase
        # a workaround for minic tf's same mode in conv2d
        # kernel_size can only be 5 or 3
        if kernel_size == 5:
            # kernel_size = 5, stride = 2
            # kernel_size = 5, stride = 1
            padding = 2
        else:
            # kernel_size = 3, stride = 1
            # kernel_size = 3, stride = 2
            padding = 1

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, kernel_size, stride=stride, padding=padding, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t, momentum=m, eps=eps),
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
            nn.BatchNorm2d(out_channels, momentum=m, eps=eps),
            Swish()
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=m, eps=eps),
            )

    def survival(self):
        var = torch.bernoulli(torch.tensor(self.sp).float())
        return torch.equal(var, torch.tensor(1).float().to(var.device))

    def se(self, x):
        squeezed = self.squeeze(x)
        squeezed = squeezed.view(squeezed.size(0), -1)
        excitation = self.excitation(squeezed)
        excitation = excitation.view(x.size(0), x.size(1), 1, 1)
        x = x * excitation
        return x

    def drop_connect(self, x):
        # this implementation(drop out certain features) is way
        # more stable than drop out the entire block during training
        # and gives a better result.
        # ""since lower earlier layers extract low-level features
        # that will be used in later layres and should therefore
        # be more reliably present."" --'Deep Networks with Stochastic
        # Depth'
        if not self.training:
            return x

        random_tensor = self.sp + torch.rand(x.size(0), 1, 1, 1)
        binary_tensor = torch.floor(random_tensor).to(x.device)

        x = x / self.sp * binary_tensor
        return x

    def residual(self, x):
        #print(x.shape)
        # expansion

        expansion = self.expansion(x)
        depthwise = self.depthwise(expansion)
        #print(depthwise.shape)
        se = self.se(depthwise)
        #print(se.shape)
        pointwise = self.pointwise(se)

        x = self.drop_connect(pointwise)
        return x
        #print(pointwise.shape)
        #return pointwise

    #@torch.jit.script_method
    def forward(self, x):

        #if self.training:
        #    if self.survival():
        #        x = self.shortcut(x) + self.residual(x)
        #    else:
        #        x = self.shortcut(x)

        #else:
        #    x = self.residual(x) * self.sp + self.shortcut(x)
        #return x
        return self.residual(x) + self.shortcut(x)

        # expansion
        #expansion = self.expansion(x)

        ## depthwise
        #depthwise = self.depthwise(expansion)

        ## squeeze and excitation
        #squeezed = self.squeeze(depthwise)
        #squeezed = squeezed.view(squeezed.size(0), -1)
        #excitation = self.excitation(squeezed)
        #excitation = excitation.view(depthwise.size(0), depthwise.size(1), 1, 1)
        #depthwise = depthwise * excitation

        ## pointwise
        #pointwise = self.pointwise(depthwise)

        # stochastic depth
        #if self.training:
        #    if self.survival():
        #        x = shortcut + pointwise
        #    else:
        #        x = shortcut

        #else:
        #    x = pointwise * self.sp + shortcut
        #residual = self.residual(x)
        #print(residual.shape)
        #print('shortcut', shortcut.shape)

        #x = shortcut + residual
        #print('x', x.shape)
        #x = shortcut + pointwise
        #residual = self.residual(x)
        #x = residual + shortcut
        #return x


class EfficientNet(nn.Module):
    def __init__(self, stage_args, net_args, num_classes=100):
        super().__init__()

        self.stage_args = stage_args
        self.net_args = net_args

        self.in_channels = 3
        out_channels = round_filters(32, self.net_args)

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, 3, stride=2),
            nn.BatchNorm2d(out_channels)
        )

        self.in_channels = out_channels

        self.current_block_id = 0
        self.conv1 = self._make_stage(0)
        self.conv2 = self._make_stage(1)
        self.conv3 = self._make_stage(2)
        self.conv4 = self._make_stage(3)
        self.conv5 = self._make_stage(4)
        self.conv6 = self._make_stage(5)
        self.conv7 = self._make_stage(6)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.net_args.dropout_rate)
        self.fc = nn.Linear(self.in_channels, num_classes)


    def total_blocks_num(self):
        count = 0
        for stage_arg in self.stage_args:
            count += stage_arg.num_repeat

        return count


    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _make_stage(self, stage_id):

        stage_args = self.stage_args[stage_id]
        out_channels = round_filters(stage_args.output_filters, self.net_args)
        kernel_size = stage_args.kernel_size
        stride = stage_args.stride
        t = stage_args.expand_ratio
        r = stage_args.se_ratio
        num_repeat = round_repeats(stage_args.num_repeat, self.net_args)

        total = self.total_blocks_num()
        sp = self.net_args.drop_connect_rate * self.current_block_id / (total - 1)
        self.current_block_id += 1
        m = self.net_args.batch_norm_momentum
        esp = self.net_args.batch_norm_epsilon

        layers = []
        strides = [stride] + [1] * (num_repeat - 1)

        for stride in strides:
            layers.append(MBConvBlock(self.in_channels, out_channels, kernel_size, stride, t, r, sp, m, esp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

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


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    # make sure that new_filters is a integer multiples of 8
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))

class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            StageArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return StageArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.
        Args:
            block (namedtuple): A BlockArgs type argument.
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """Create BlockArgs and GlobalParams for efficientnet model.
    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)
        Meaning as the name suggests.
    Returns:
        blocks_args, global_params.
    """

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = NetArgs(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,

        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, global_params

def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.
    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.
    Returns:
        blocks_args, global_params
    """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)

    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)

    #print('net:',global_params)
    return blocks_args, global_params


def efficientnetb0(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b0', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb1(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b1', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb2(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b2', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb3(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b3', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb4(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b4', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb5(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b5', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb6(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b6', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetb7(num_classes):
    stage_args, net_args = get_model_params('efficientnet-b7', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

def efficientnetl2(num_classes):
    stage_args, net_args = get_model_params('efficientnet-l2', {})
    return EfficientNet(stage_args, net_args, num_classes=num_classes)

#efficientnetb0()
#efficientnetb1()
#efficientnetb2()


#net = efficientnetb0(37)

#t = torch.Tensor(256, 3, 224, 224)
#print(net(t).shape)
#print(sum([p.numel() for p in net.parameters()]))