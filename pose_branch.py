import torch
import torch.nn as nn

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.conv1 = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.conv2 = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv4 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        cat_features = torch.cat([self.relu(self.conv1(f)) for f in last_features], dim=1)

        out = self.relu(self.conv2(cat_features))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)

        out = out.mean(dim=[2,3])

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
