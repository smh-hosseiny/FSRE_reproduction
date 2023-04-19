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
        self.opt = options
        self.epoch = 0
        
        self.models = nn.ModuleDict({
            'encoder': ResnetEncoder(num_layers=self.opt.num_layers, pretrained=self.opt.pretrained),
            'pose_encoder': ResnetEncoder(num_layers=18, num_input_images=2, pretrained=self.opt.pretrained),
            'pose': PoseDecoder(ResnetEncoder(num_layers=18, num_input_images=2, pretrained=self.opt.pretrained).num_ch_enc)
        })
        
        if not self.opt.no_cma:
            self.models.update({
                'decoder': CMA(self.models['encoder'].num_ch_enc, opt=self.opt)
            })
        else:
            self.models.update({
                'depth': DepthDecoder(self.models['encoder'].num_ch_enc, scales=self.opt.scales, opt=self.opt),
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
    

    def compute_outputs(self, inputs):
        outputs = {}
        features = {}
        center = inputs["color_aug", 0, 0].cuda()

        with torch.no_grad():
            features[0] = self.models["encoder"](center)
            for frame_id in self.opt.frame_ids[1:]:
                color_aug = inputs["color_aug", frame_id, 0].cuda()

                if frame_id == 1:
                    pose_inputs = torch.cat([center, color_aug], dim=1)
                elif frame_id == -1:
                    pose_inputs = torch.cat([color_aug, center], dim=1)
                else:
                    raise Exception("invalid frame_ids")

                if pose_inputs.shape[3] > 640:
                    pose_inputs = F.interpolate(pose_inputs, size=(192, 640), mode='bilinear')
                pose_features = self.models['pose_encoder'](pose_inputs)
                axisangle, translation = self.models['pose']([pose_features])
                outputs["axisangle", frame_id] = axisangle
                T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=frame_id < 0)
                outputs["T", frame_id] = T

        if not self.opt.no_cma:
            disp, seg = self.models['decoder'](features[0])
            outputs.update(disp)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs["disp", s], (self.opt.height, self.opt.width), mode='bilinear', align_corners=False)
                else:
                    disp = outputs["disp", s]
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs["depth", 0, s] = depth
            outputs.update(seg)
        else:
            if self.opt.semantic_distil is not None:
                seg = self.models["seg"](features[0])
                outputs.update(seg)

            outputs.update(self.models["depth"](features[0]))
            _, depth = disp_to_depth(outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs["disp", s], (self.opt.height, self.opt.width), mode='bilinear',
                                        align_corners=False)
                else:
                    disp = outputs["disp", s]
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs["depth", 0, s] = depth

        for frame_id in self.opt.frame_ids[1:]:
            for s in self.opt.scales:
                cam_points = self.backproject_depth(outputs["depth", 0, s], inputs["inv_K"].cuda())
                pix_coords, next_depth = self.project_3d(cam_points, inputs["K"].cuda(), outputs["T", frame_id])
                outputs["sample", frame_id, s] = pix_coords
                outputs["next_depth", frame_id, s] = next_depth

        return outputs
    


def disp_to_depth(disp, min_depth, max_depth):
    """Convert seg_networks's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

    return scaled_disp, depth

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the seg_networks's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

