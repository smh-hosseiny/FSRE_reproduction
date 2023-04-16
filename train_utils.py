import torch
import torch.nn as nn
import torch.nn.functional as F

from CMA import CMA
from depth_branch import DepthDecoder
from pose_branch import PoseDecoder
from resnet_backbone import ResnetEncoder
from segmentation_branch import Segmentor
from utils import *


class TrainerParallel(nn.Module):
    def __init__(self, options):
        super().__init__()

        # Initialize variables and models
        self.opt = options
        self.epoch = 0

        self.models = nn.ModuleDict({
            'encoder': ResnetEncoder(num_layers=self.opt.num_layers, pretrained=self.opt.pretrained),
            'pose_encoder': ResnetEncoder(num_layers=18, num_input_images=2, pretrained=self.opt.pretrained),
            'pose': PoseDecoder(self.models['pose_encoder'].num_ch_enc)
        })

        # Add depth decoder or CMA
        if not self.opt.no_cma:
            self.models.update({'decoder': CMA(self.models['encoder'].num_ch_enc, opt=self.opt)})
        else:
            self.models.update({
                'depth': DepthDecoder(self.models['encoder'].num_ch_enc, scales=self.opt.scales, opt=self.opt)
            })
            if self.opt.semantic_distil is not None:
                self.models['seg'] = Segmentor(self.models['encoder'].num_ch_enc, scales=[0])

        # Initialize projection functions
        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)

        # Initialize loss functions and masking functions
        self.loss_functions = {self.compute_reprojection: self.opt.reprojection}
        self.masking_functions = []

        if self.opt.disparity_smoothness:
            self.loss_functions[self.compute_smoothness] = self.opt.disparity_smoothness

        if self.opt.semantic_distil:
            self.loss_functions[self.compute_semantic_distil] = self.opt.semantic_distil

        if self.opt.sgt:
            self.loss_functions[self.compute_sgt_loss] = self.opt.sgt

        # Initialize parameters to train
        self.parameters_to_train = []
        for model in self.models:
            self.parameters_to_train += list(self.models[model].parameters())

        # Initialize additional variables
        self.ssim = SSIM()

    def forward(self, inputs):
        outputs = self.compute_outputs(inputs)
        losses = {loss_function.__name__: loss_function(inputs, outputs) * loss_weight
                for loss_function, loss_weight in self.loss_functions.items()}
        loss = sum(losses.values())
        losses["loss"] = loss
        for key, value in outputs.items():
            if key != 'loss':
                outputs[key] = value.detach()
        return losses, outputs




def compute_sgt_loss(self, inputs, outputs):

    assert len(self.opt.sgt_layers) == len(self.opt.sgt_kernel_size)

    total_loss = 0
    for s, kernel_size in zip(self.opt.sgt_layers, self.opt.sgt_kernel_size):
        pad = kernel_size // 2
        h = self.opt.height // 2 ** s
        w = self.opt.width // 2 ** s
        seg_target = inputs[("seg", 0, s)]
        seg = F.interpolate(seg_target, size=(h, w), mode='nearest')
        center = seg
        padded = F.pad(center, [pad] * 4, value=-1)
        aggregated_label = torch.zeros(*(center.shape + (kernel_size, kernel_size))).to(center.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted = padded[:, :, 0 + i: h + i, 0 + j:w + j]
                label = center == shifted
                aggregated_label[:, :, :, :, i, j] = label
        aggregated_label = aggregated_label.float()
        pos_idx = (aggregated_label == 1).float()
        neg_idx = (aggregated_label == 0).float()
        pos_idx_num = pos_idx.sum(dim=-1).sum(dim=-1)
        neg_idx_num = neg_idx.sum(dim=-1).sum(dim=-1)

        boundary_region = (pos_idx_num >= kernel_size - 1) & (neg_idx_num >= kernel_size - 1)
        non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0)

        if s == min(self.opt.sgt_layers):
            outputs[('boundary', s)] = boundary_region.data
            outputs[('non_boundary', s)] = non_boundary_region.data

        feature = outputs[('d_feature', s)]
        affinity = self.compute_affinity(feature, kernel_size=kernel_size)
        pos_dist = (pos_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / pos_idx.sum(dim=-1).sum(dim=-1)[boundary_region]
        neg_dist = (neg_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / neg_idx.sum(dim=-1).sum(dim=-1)[boundary_region]
        zeros = torch.zeros(pos_dist.shape, device=pos_dist.device)
        loss = torch.max(zeros, pos_dist - neg_dist + self.opt.sgt_margin)

        total_loss += loss.mean() / (2 ** s)

    return total_loss




def compute_reprojection(self, inputs, outputs):
    target = inputs[("color", 0, 0)]
    total_losses = 0

    for s in self.opt.scales:
        losses = []
        identity_reprojection_losses = []

        for frame_id in self.opt.frame_ids[1:]:
            pred = F.grid_sample(inputs[("color", frame_id, 0)], outputs[("sample", frame_id, s)],
                                 padding_mode="border", align_corners=True)
            outputs[("color", frame_id, s)] = pred

            reprojection_loss = self.reprojection_loss(pred, target)
            outputs[("reprojection_loss", frame_id)] = reprojection_loss

            losses.append(reprojection_loss)

            pred = inputs[("color", frame_id, 0)]
            identity_reprojection_losses.append(self.reprojection_loss(pred, target))

        losses = torch.cat(losses, dim=1)

        # Apply automask in Monodepth2
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, dim=1)
        identity_reprojection_losses += torch.randn_like(identity_reprojection_losses) * 0.00001
        combined = torch.cat([losses, identity_reprojection_losses], dim=1)

        to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)
        total_losses += to_optimise.mean() / (2 ** s)

    return total_losses

 
def compute_semantic_distil(self, inputs, outputs):
    seg_target = inputs[("seg", 0, 0)].long().squeeze(1)
    seg_pred = outputs[("seg_logits", 0)]
    weights = seg_target.sum(1, keepdim=True).float()
    ignore_mask = (weights == 0)
    weights[ignore_mask] = 1
    seg_loss = F.cross_entropy(seg_pred, seg_target, reduction='none')
    total_loss = seg_loss.mean()
    return total_loss



def compute_smoothness(self, inputs, outputs):
    total_loss = 0
    for s in self.opt.scales:
        disp = outputs[("disp", s)]
        color = inputs[("color", 0, s)]
        mean_disp = disp.mean(dim=(2, 3), keepdim=True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)
        total_loss += smooth_loss / (2 ** s)

    return total_loss

def reprojection_loss(self, pred, target):
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(dim=1, keepdim=True)
    ssim_loss = self.ssim(pred, target).mean(dim=1, keepdim=True)
    loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return loss

def compute_affinity(self, feature, kernel_size):
    pad = kernel_size // 2
    feature = F.normalize(feature, dim=1)
    unfolded = F.unfold(F.pad(feature, [pad] * 4), kernel_size, 1, pad).view(feature.size(0), -1, kernel_size * kernel_size, feature.size(2), feature.size(3))
    feature = feature.unsqueeze(2).unsqueeze(3)
    similarity = (feature * unfolded).sum(dim=1, keepdim=True)
    eps = torch.full_like(similarity, 1e-9)
    affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
    return affinity

