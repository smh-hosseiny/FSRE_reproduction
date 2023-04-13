import numpy as np
from utils import *

class Segmentor(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=19):
        super(Segmentor, self).init()
        self.num_output_channels = num_output_channels
        self.skip_conc = True
        self.scales = [0]
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.decoder = nn.ModuleList()

        num_ch_in = self.num_ch_enc[-1]
        for i in range(4, -1, -1):
            num_ch_out = self.num_ch_dec[i]
            self.decoder.append(ConvBlock(num_ch_in, num_ch_out))
            if self.skip_conc and i > 0:
                num_ch_in = num_ch_out + self.num_ch_enc[i - 1]
            else:
                num_ch_in = num_ch_out
            if i in self.scales:
                self.decoder.append(Conv3x3(num_ch_out, self.num_output_channels))
        self.decoder.reverse()




    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]

        for i in range(5):
            x = self.decoder[i*2](x)

            if i > 0 and self.skip_conc:
                x = torch.cat([upsample(x), input_features[-i]], dim=1)
            else:
                x = upsample(x)

            x = self.decoder[i*2+1](x)
            outputs[('s_feature', 4 - i)] = x

            if 4 - i in self.scales:
                out = self.decoder[10 + 4 - i](x)
                outputs[('seg_logits', 4 - i)] = out

        return outputs

