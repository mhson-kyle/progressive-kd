import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from networks.resnet import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

# Encoder Used in Transformer Layer
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, vis):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vis = vis
        
        self.num_attention_heads = self.num_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(0) # config.transformer["attention_dropout_rate"] 
        self.proj_dropout = nn.Dropout(0) # config.transformer["attention_dropout_rate"]

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

# MLP Layer Used in Transformer Layer
class Mlp(nn.Module):
    def __init__(self, mlp_dim, hidden_size):
        super(Mlp, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.hidden_size)
        self.act_fn = nn.functional.gelu # ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

## Convolutional Embedding use ResNET
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, resnet_num_layers=(3, 4, 9), resnet_width_factor=1, grid_size=(8, 8), hidden_size=768, img_size=128, in_channels=3):
        super(Embeddings, self).__init__()
        self.resnet_num_layers = resnet_num_layers
        self.resnet_width_factor = resnet_width_factor
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.img_size = _pair(img_size)
        
        patch_size = (self.img_size[0] // 16 // self.grid_size[0], self.img_size[1] // 16 // self.grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (self.img_size[0] // patch_size_real[0]) * (self.img_size[1] // patch_size_real[1])
        self.conv2d = Conv2d(in_channels=32,
                            out_channels=3,
                            kernel_size=1,
                            stride=1)
        self.CNN_Model = ResNetV2(block_units=self.resnet_num_layers, width_factor=self.resnet_width_factor)
        in_channels = self.CNN_Model.width * 16
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=self.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_size))
        self.dropout = nn.Dropout(0.1) # config.transformer["dropout_rate"]


    def forward(self, x):
        x = self.conv2d(x)
        x, features = self.CNN_Model(x) # features : feature maps from resnet
        x = self.patch_embeddings(x)  # (Batch, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2) # (Batch, hidden, n_patches)
        x = x.transpose(-1, -2)  # (Batch, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

## Single Transformer Layer
class Block(nn.Module):
    def __init__(self, mlp_dim=3072, hidden_size=768, num_heads=12, vis=False):
        super(Block, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vis = vis
        
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(self.mlp_dim, self.hidden_size)
        self.attn = Attention(self.hidden_size, self.num_heads, self.vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

## Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, num_layers, mlp_dim, hidden_size, num_heads, vis):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vis = vis
        
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(self.hidden_size, eps=1e-6)
        for _ in range(self.num_layers):
            layer = Block(self.mlp_dim, self.hidden_size, self.num_heads, self.vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, embedding_output):
        attn_weights = []
        for layer_block in self.layer:
            embedding_output, weights = layer_block(embedding_output)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(embedding_output)
        return encoded, attn_weights

## Embedding + Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, 
                 resnet_num_layers, 
                 resnet_width_factor, 
                 grid_size, 
                 num_layers, 
                 mlp_dim, 
                 hidden_size, 
                 img_size, 
                 num_heads, 
                 vis):
        super(Transformer, self).__init__()
        self.resnet_num_layers = resnet_num_layers
        self.resnet_width_factor = resnet_width_factor
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.num_heads = num_heads
        self.vis = vis
        
        self.embeddings = Embeddings(resnet_num_layers=self.resnet_num_layers, 
                                    resnet_width_factor=self.resnet_width_factor, 
                                    grid_size=self.grid_size, 
                                    hidden_size=self.hidden_size, 
                                    img_size=self.img_size)
        self.encoder = Encoder(num_layers=self.num_layers, 
                               mlp_dim=self.mlp_dim, 
                               hidden_size=self.hidden_size, 
                               num_heads=self.num_heads, 
                               vis=self.vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

# Used in Decoder Block
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

# Decoder
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up = nn.UpsamplingBinn.linear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Segmentation Head
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBinn.linear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# Total Decoder
class DecoderCup(nn.Module):
    def __init__(self, decoder_channels=(256, 128, 64, 16), n_skip=3, skip_channels=[512, 256, 64, 16], hidden_size=768):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.n_skip = n_skip
        self.skip_channels = skip_channels
        self.hidden_size = hidden_size
        
        self.head_channels = 512
        self.conv_more = Conv2dReLU(
            self.hidden_size,
            self.head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        self.in_channels = [self.head_channels] + list(self.decoder_channels[:-1])
        self.out_channels = self.decoder_channels
        self.out_conv2d_list = nn.Conv2d(self.out_channels[-1], 1, 1, padding=0) 
        if self.n_skip != 0:
            self.skip_channels = skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                self.skip_channels[3-i]=0

        else:
            self.skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        self.hidden_states = features[0]
        features = features[1:]
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        x = self.out_conv2d_list(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, 
                 resnet_num_layers, resnet_width_factor, grid_size, num_layers, 
                 mlp_dim, hidden_size, num_heads, 
                 decoder_channels=(256, 128, 64, 16), n_skip=3, skip_channels=[512, 256, 64, 16], img_size=128, n_classes=2, 
                 zero_head=False, vis=False):
        
        super(VisionTransformer, self).__init__()
        self.resnet_num_layers = resnet_num_layers
        self.resnet_width_factor = resnet_width_factor
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.num_heads = num_heads
        self.n_skip = n_skip
        self.decoder_channels = decoder_channels
        self.skip_channels = skip_channels
        self.n_classes = n_classes
        self.vis = vis
        self.zero_head = zero_head
        self.classifier = 'seg'
        
        self.transformer = Transformer(resnet_num_layers=self.resnet_num_layers, 
                                    resnet_width_factor=self.resnet_width_factor, 
                                    grid_size=self.grid_size, 
                                    num_layers=self.num_layers, 
                                    mlp_dim=self.mlp_dim, 
                                    hidden_size=self.hidden_size, 
                                    img_size=self.img_size, 
                                    num_heads=self.num_heads, 
                                    vis=self.vis
                                    )
        self.decoder_1 = DecoderCup(decoder_channels=self.decoder_channels, 
                                  n_skip=self.n_skip, 
                                  skip_channels=self.skip_channels, 
                                  hidden_size=self.hidden_size)
        self.decoder_2 = DecoderCup(decoder_channels=self.decoder_channels, 
                                  n_skip=self.n_skip, 
                                  skip_channels=self.skip_channels, 
                                  hidden_size=self.hidden_size)
        self.decoder_3 = DecoderCup(decoder_channels=self.decoder_channels, 
                                  n_skip=self.n_skip, 
                                  skip_channels=self.skip_channels, 
                                  hidden_size=self.hidden_size)
        self.map_out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        bottleneck_feature, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        
        out_map_tmax = self.map_out_activation(self.decoder_1(bottleneck_feature, features))
        out_map_cbv = self.map_out_activation(self.decoder_2(bottleneck_feature, features))
        out_map_cbf = self.map_out_activation(self.decoder_3(bottleneck_feature, features))
        
        map_pred_dict = {
            'out_map_tmax' : out_map_tmax,
            'out_map_cbv' : out_map_cbv,
            'out_map_cbf' : out_map_cbf
        }
        return map_pred_dict, features, bottleneck_feature

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:] # 
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            self.transformer.embeddings.CNN_Model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.transformer.embeddings.CNN_Model.root.gn.weight.copy_(gn_weight)
            self.transformer.embeddings.CNN_Model.root.gn.bias.copy_(gn_bias)
            #
            for bname, block in self.transformer.embeddings.CNN_Model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)     