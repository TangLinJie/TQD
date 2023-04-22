from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
BatchNorm2d = nn.BatchNorm2d

from .transformer import TransformerDecoder, TransformerDecoderLayer
from .position_encoding import PositionEmbeddingSine

class TextQuerySegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 num_queries=5, num_query_fusing_layer=6, dim_feedforward=2048,
                 deep_supervision=False, kernel_size=[1,1],
                 *args, **kwargs):
        super(TextQuerySegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.query_embed = nn.Embedding(num_queries, inner_channels)

        self.global_feature_reduction = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        query_fusing_decoder_layer = TransformerDecoderLayer(
            d_model=inner_channels,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            normalize_before=False,
        )
        self.query_fusing_decoder = TransformerDecoder(
            query_fusing_decoder_layer, num_query_fusing_layer, return_intermediate=deep_supervision
        )

        self.pe_layer = PositionEmbeddingSine(inner_channels//2, normalize=True)

        self.kernel_size = kernel_size
        self.kernel_pred = nn.Linear(inner_channels, inner_channels//4*self.kernel_size[0]*self.kernel_size[1])

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels//4,
                      3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True))
        self.feature_upsample.apply(self.weights_init)
        self.inner_channels = inner_channels

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)
        self.global_feature_reduction.apply(self.weights_init)
        self._reset_query_fusing_encoder_parameters(self.query_fusing_decoder)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _reset_query_fusing_encoder_parameters(self, module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)

        # for getting the same size with input (stride: 1)
        assert (self.kernel_size[0] - 1) % 2 == 0
        assert (self.kernel_size[1] - 1) % 2 == 0
        h_padding = int((self.kernel_size[0] - 1.) // 2)
        w_padding = int((self.kernel_size[1] - 1.) // 2)

        fuse = self.feature_upsample(fuse)

        bs = c5.shape[0]
        global_feature = self.global_feature_reduction(c5)
        global_feature_pe = self.pe_layer(global_feature)
        global_feature = global_feature.flatten(2).permute(2, 0, 1) # hw b c
        global_feature_pe = global_feature_pe.flatten(2).permute(2, 0, 1)
        batch_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(batch_query_embed)
        global_query_embedding, _ = self.query_fusing_decoder(
            tgt, global_feature, pos=global_feature_pe, query_pos=batch_query_embed
        )
        global_query_embedding = global_query_embedding.transpose(1, 2) # l b q c
        kernel_params = self.kernel_pred(global_query_embedding)
        layer_masks = []
        for layer_idx in range(kernel_params.shape[0]):
            batch_masks = []
            for img_idx in range(kernel_params.shape[1]):
                img_query_masks = []
                for query_idx in range(kernel_params.shape[2]):
                    query_mask = F.conv2d(fuse[img_idx].unsqueeze(0), kernel_params[layer_idx][img_idx][query_idx].view(\
                                            1, self.inner_channels//4, self.kernel_size[0], self.kernel_size[1]),\
                                            stride=1, padding=(h_padding, w_padding)).view(1, 1, fuse.shape[2], fuse.shape[3])
                    img_query_masks.append(query_mask)
                img_query_masks = torch.cat(img_query_masks, dim=1).max(dim=1)[0]
                batch_masks.append(img_query_masks)
            batch_masks = torch.cat(batch_masks)
            layer_masks.append(batch_masks)
        layer_masks = torch.stack(layer_masks) # l b h w
        binary = F.sigmoid(layer_masks).unsqueeze(2) # l b 1 h w
        if self.training:
            result = OrderedDict(binary=binary)
            return result
        else:
            return binary[-1]