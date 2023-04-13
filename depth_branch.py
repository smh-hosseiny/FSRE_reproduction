import torch.nn.functional as F
from utils import *

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(DepthDecoder, self).init()    
        self.opt = None
        self.skip_conc = True
        self.scales = range(4)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.upsample_mode = 'nearest'
        self.num_ch_enc = num_ch_enc

        convs = [ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[4])]  

        for i in range(3, -1, -1):
            num_ch_in = self.num_ch_enc[i] if self.use_skips else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            convs.extend([ConvBlock(num_ch_in, num_ch_out), ConvBlock(num_ch_out, num_ch_out)])

        for s in self.scales:
            convs.append(Conv3x3(self.num_ch_dec[s], 1))

        self.decoder = nn.ModuleList(convs)
        self.sigmoid = nn.Sigmoid()


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.decoder[-2*i](x)
            if self.skip_conc and i > 0:
                x = torch.cat([upsample(x), input_features[i-1]], dim=1)
            else:
                x = upsample(x)
            x = self.decoder[-2*i+1](x)
            outputs[('s_feature', i)] = x
            if i in self.scales:
                outputs[('seg_logits', i)] = self.decoder[10+i](x)
        
        return outputs

