from utils import *
from segmentation_branch import Segmentor
from depth_branch import DepthDecoder


def conv_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class CMA(nn.Module):
    def __init__(self, num_ch_enc=None, opt=None):
        super(CMA, self).__init__()

        self.scales = opt.scales
        cma_layers = opt.cma_layers
        self.opt = opt
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        in_channels_list = [32, 64, 128, 256, 16]

        num_output_channels = 1

        self.depth_decoder = DepthDecoder(num_ch_enc, num_output_channels=num_output_channels,
                                          scales=opt.scales,
                                          opt=self.opt)
        self.seg_decoder = Segmentor(num_ch_enc, num_output_channels=19,
                                      scales=[0])

        self.att_d_to_s = nn.ModuleDict(
            {str(i): MultiEmbedding(in_channels=in_channels_list[i], num_head=opt.num_head, ratio=opt.head_ratio) for i in cma_layers}
        )

        self.att_s_to_d = nn.ModuleDict(
            {str(i): MultiEmbedding(in_channels=in_channels_list[i], num_head=opt.num_head, ratio=opt.head_ratio) for i in cma_layers}
        )

    def forward(self, input_features):
        depth_outputs, seg_outputs = {}, {}
        x_d, x_s = None, None
        for i in reversed(range(5)):
            x = input_features[i]
            x_d = self.depth_decoder.decoder[-2 * i + 8](x_d or x)
            x_s = self.seg_decoder.decoder[-2 * i + 8](x_s or x)
            x_d, x_s = upsample(x_d), upsample(x_s)
            if i > 0:
                x_d, x_s = torch.cat([x_d, input_features[i - 1]]), torch.cat([x_s, input_features[i - 1]])
            x_d, x_s = self.depth_decoder.decoder[-2 * i + 9](x_d), self.seg_decoder.decoder[-2 * i + 9](x_s)
            if i - 1 in self.opt.cma_layers:
                attn = str(i - 1)
                x_d_att = self.att_d_to_s[attn](x_d, x_s)
                x_s_att = self.att_s_to_d[attn](x_s, x_d)
                x_d, x_s = x_d_att, x_s_att
            if self.opt.sgt:
                depth_outputs[('d_feature', i)] = x_d
                seg_outputs[('s_feature', i)] = x_s
            if i in self.scales:
                depth_outs = self.depth_decoder.decoder[10 + i](x_d)
                depth_outputs[("disp", i)] = F.sigmoid(depth_outs[:, :1, :, :])
                if i == 0:
                    seg_outs = self.seg_decoder.decoder[10 + i](x_s)
                    seg_outputs[("seg_logits", i)] = seg_outs[:, :19, :, :]
        return depth_outputs, seg_outputs



class MultiEmbedding(nn.Module):
    def __init__(self, in_channels, num_head, ratio):
        super(MultiEmbedding, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.out_channel = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.W = conv_bn(int(in_channels * ratio), in_channels)
        self.fuse = nn.Sequential(ConvBlock(in_channels * 2, in_channels), nn.Conv2d(in_channels, in_channels, kernel_size=1))

    def forward(self, key, query):
        q_out = self.query_conv(query).view(query.size(0), self.num_head, self.out_channel, -1).transpose(2, 3)
        k_out = self.key_conv(key).view(key.size(0), self.num_head, self.out_channel, -1)
        v_out = self.value_conv(key).view(key.size(0), self.num_head, self.out_channel, -1)

        att = torch.matmul(q_out, k_out) / np.sqrt(self.out_channel)

        if self.num_head == 1:
            softmax = F.softmax(att, dim=2)
        else:
            softmax = F.softmax(att, dim=3)

        weighted_value = torch.matmul(softmax, v_out.transpose(2, 3))
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(query.size(0), -1, query.size(2), query.size(3))
        out = self.conv_bn(weighted_value)

        return self.fuse(torch.cat([key, out], dim=1))

