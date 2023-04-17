import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.kitti_dataset import KittiDataset
from CMA import CMA
from depth_branch import DepthDecoder
from resnet_backbone import ResnetEncoder
from options import Options
from utils import readlines, disp_to_depth

cv2.setNumThreads(0)

# Set path to the folder containing the split files
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Set the stereo scale factor to convert predicted depths to real-world scale
STEREO_SCALE_FACTOR = 5.4

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            f"Cannot find a folder at {opt.load_weights_folder}"
        print(f"-> Loading weights from {opt.load_weights_folder}")
        
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        dataset = KittiDataset(height=encoder_dict['height'], width=encoder_dict['width'],
                               frame_idxs=[0], filenames=filenames, data_path=opt.data_path, is_train=False,
                               num_scales=len(opt.scales))
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=12,
                                pin_memory=True, drop_last=False)
        encoder = ResnetEncoder(num_layers=opt.num_layers)
        depth_decoder = CMA(encoder.num_ch_enc, opt=opt) if not opt.no_cma else \
                        DepthDecoder(encoder.num_ch_enc, scales=opt.scales, opt=opt)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder, depth_decoder = encoder.cuda(), depth_decoder.cuda()
        encoder.eval(), depth_decoder.eval()
        
        pred_disps, models = [], {'encoder': encoder, 'depth': depth_decoder}
        print(f"-> Computing predictions with size {encoder_dict['width']}x{encoder_dict['height']}")
        with torch.no_grad():
            for data in dataloader:
                data = {key: val.cuda() for key, val in data.items()}
                input_color = data[("color", 0, 0)]
                features = models['encoder'](input_color)
                output = models['depth'](features)[0] if not opt.no_cma else models['depth'](features)
                pred_disp = output[("disp", 0)]
                pred_disp, _ = disp_to_depth(pred_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps)
    else:
        # Load predictions from file
        print(f"-> Loading predictions from {opt.ext_disp_to_eval}")
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return

    if opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print(f"-> Saving out benchmark predictions to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        for idx, pred_disp in enumerate(pred_disps):
            depth = 32.779243 / cv2.resize(pred_disp, (1216, 352))
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            cv2.imwrite(os.path.join(save_dir, f"{idx:010d}.png"), depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    splits_dir = os.path.join(opt.data_path, "splits")
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    print("-> Evaluating")

    if opt.eval_stereo:
        print(f"   Stereo evaluation - disabling median scaling, scaling by {STEREO_SCALE_FACTOR}")
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i, pred_disp in enumerate(pred_disps):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH, out=pred_depth)

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")
        print(med, np.mean(ratios))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


if __name__ == "__main__": 
    evaluate(Options().parse())

