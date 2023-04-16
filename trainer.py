from datetime import datetime
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from utils import *


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, datetime.now().strftime("%m-%d-%Y"), self.opt.model_name)

        # Checking height and width are multiples of 32
        assert self.opt.height % 32 == 0 and self.opt.width % 32 == 0, "'height' and 'width' must be multiples of 32"

        self.trainer = TrainerParallel(options)
        self.model_optimizer = optim.Adam(self.trainer.parameters_to_train, self.opt.learning_rate)
        self.epoch, self.step, self.is_best, self.best_loss = 0, 0, False, 1e9

        if self.opt.load_weights_folder:
            self.load_model()

        print(f"Training model named:\n{self.opt.model_name}\nModels and tensorboard events files are saved to:\n{self.opt.log_dir}")

        # Data
        train_filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt".format("train")))
        train_dataset = KittiDataset(height=self.opt.height, width=self.opt.width, frame_idxs=self.opt.frame_ids, 
                filenames=train_filenames, data_path=self.opt.data_path, is_train=True, num_scales=len(self.opt.scales))


        val_filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt".format("val")))
        val_dataset = KittiDataset(height=self.opt.height, width=self.opt.width, frame_idxs=self.opt.frame_ids, 
                filenames=val_filenames, data_path=self.opt.data_path, is_train=False, num_scales=len(self.opt.scales))

        if self.opt.local_rank == 0:
            self.writers = {mode: SummaryWriter(os.path.join(self.log_path, mode)) for mode in ["train", "val"]}

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print(f"Using split:\n{self.opt.split}\nThere are {len(train_dataset)} training items and {len(val_dataset)} validation items\n")

        torch.cuda.set_device(self.opt.local_rank)
        self.trainer = self.trainer.cuda()

        train_sampler, val_sampler = RandomSampler(train_dataset), SequentialSampler(val_dataset)

        self.train_loader, self.val_loader = [
            DataLoader(
                dataset,
                self.opt.batch_size,
                shuffle = dataset == train_dataset,
                num_workers=self.opt.batch_size,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
            ) for dataset, sampler in zip([train_dataset, val_dataset], [train_sampler, val_sampler])
        ]

        

    def lr_decay(self):
        print('Decaying the learning rate...')
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] *= self.opt.decay_rate

    def set_train(self):
        self.trainer.train()

    def set_eval(self):
        self.trainer.eval()

    def train(self):
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            if self.epoch in self.opt.lr_decay:
                self.lr_decay()
            self.run_epoch()
            self.validate()

            if self.opt.local_rank == 0:
                self.save_model()

     
            self.trainer.epoch += 1

    def run_epoch(self):
        print("Training")
        self.set_train()
        data_loading_time = 0
        gpu_time = 0
        before_op_time = time.time()
        train_loss = None

        for batch_idx, inputs in enumerate(self.train_loader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()

            losses, outputs = self.trainer(inputs)
            losses['loss'] = losses['loss'].mean()

            if self.opt.split != 'test':
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time
            gpu_time += duration

            for loss_type in losses:
                losses[loss_type] = torch.mean(losses[loss_type])

            if self.opt.local_rank == 0 and (batch_idx % 250 == 0 and self.step < 2000 or self.step % 2000 == 0):
                self.log_time(batch_idx, duration, losses, data_loading_time, gpu_time)
                self.log("train", inputs, outputs, {})
                data_loading_time = 0
                gpu_time = 0

            self.step += 1
            before_op_time = time.time()

            if self.opt.local_rank == 0:
                if train_loss is None:
                    train_loss = {loss_type: float(losses[loss_type].data.mean()) for loss_type in losses}
                else:
                    for loss_type in losses:
                        train_loss[loss_type] += float(losses[loss_type].data.mean())

        if self.opt.local_rank == 0:
            for key in train_loss:
                train_loss[key] /= len(self.train_loader)
            self.log("train", inputs, outputs, train_loss)



    def val(self):
        self.set_eval()
        val_loss = {}

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()

                losses, outputs = self.trainer(inputs)

                for loss_type, loss_value in losses.items():
                    if loss_type not in val_loss:
                        val_loss[loss_type] = 0.0
                    val_loss[loss_type] += losses[loss_type].mean().item()

                self.log("val", inputs, outputs, {})

        if self.opt.local_rank == 0:
            for key in val_loss:
                val_loss[key] /= len(self.val_loader)
            val_loss['loss'] = val_loss.pop('loss')
            self.log("val", {}, {}, val_loss)
            if val_loss['loss'] < self.best_loss:
                self.is_best = True
                self.best_loss = val_loss['loss']
            else:
                self.is_best = False

        del inputs, outputs, losses
        self.set_train()

    

   

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics to monitor during training"""
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt, depth_pred = depth_gt[mask], depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = losses.get(metric, 0.0) + depth_errors[i].data.cpu()




    def log_time(self, batch_idx, duration, losses, data_time, gpu_time):
        samples_per_sec = self.opt.batch_size / duration
        time_elapsed = time.time() - self.start_time
        if self.step > 0:
            training_time_left = (self.num_total_steps / self.step - 1.0) * time_elapsed
        else:
            training_time_left = 0

        log_string = (
            f"epoch {self.epoch:>3} | batch {batch_idx:>6} | examples/s: {samples_per_sec:5.1f}"
            f" | loss: {losses['loss'].data.cpu().mean():.5f}"
            f" | time elapsed: {sec_to_hm_str(time_elapsed)}"
            f" | time left: {sec_to_hm_str(training_time_left)}"
            f" | CPU/GPU time: {data_time:.1f}s/{gpu_time:.1f}s"
        )
        print(log_string)

        log_loss = {item: round(float(losses[item].data.cpu().mean()), 6) for item in losses}
        print(log_loss)


    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]

        for loss_name, loss_value in losses.items():
            writer.add_scalar(f"{loss_name}", loss_value, self.step)

        for j in range(min(3, self.opt.batch_size)):
            for frame_id in self.opt.frame_ids:
                color = inputs[("color", frame_id, j)][j]
                writer.add_image(f"color_{frame_id}_{j}/{frame_id}", color, self.step)

                if frame_id != 0:
                    color_pred = outputs[("color", frame_id, j)][j]
                    writer.add_image(f"color_pred_{frame_id}_{j}/{frame_id}", color_pred, self.step)
                else:
                    disp = outputs[("disp", j)].data[j]
                    writer.add_image(f"disp_{j}/{frame_id}", normalize_image(disp), self.step)

                    if self.opt.semantic_distil and j == 0:
                        semantic_target = inputs[("seg", 0, 0)][j].data
                        writer.add_image("semantic_target_{}".format(j), decode_seg_map(semantic_target), self.step)

                        seg_logits = outputs[("seg_logits", 0)].argmax(dim=1, keepdim=True)
                        semantic_pred = seg_logits[j].data
                        writer.add_image(f"semantic_pred_{j}", decode_seg_map(semantic_pred), self.step)

                    if self.opt.sgt:
                        layer = min(self.opt.sgt_layers)
                        boundary_region = outputs[("boundary", layer)][j]
                        non_boundary_region = outputs[("non_boundary", layer)][j]
                        writer.add_image(f"boundary_{layer}_{j}", boundary_region, self.step)
                        writer.add_image(f"non_boundary_{layer}_{j}", non_boundary_region, self.step)



    def save_model(trainer, epoch, save_dir, is_best=False):
        save_folder = os.path.join(save_dir, "models", f"weights_{epoch}" + ('_best' if is_best else ''))
        os.makedirs(save_folder, exist_ok=True)

        models = trainer.module.models if hasattr(trainer, 'module') else trainer.models

        for model_name, model in models.items():
            model_save_path = os.path.join(save_folder, f"{model_name}.pth")
            torch.save(model.state_dict(), model_save_path)

            if model_name == 'encoder':
                # save sizes, epoch and step needed at prediction time
                model_state_dict = {
                    'height': trainer.opt.height,
                    'width': trainer.opt.width,
                    'epoch': epoch,
                    'step': trainer.step,
                }
                torch.save(model_state_dict, os.path.join(save_folder, 'encoder_info.pth'))

        torch.save(trainer.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))


    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(vars(self.opt), f, indent=2)



    def load_model(self):
        folder_path = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(folder_path), "Invalid folder path: {}".format(folder_path)
        print("Loading model from folder: {}".format(folder_path))

        for model_name, model in self.trainer.models.items():
            print("Loading {} weights...".format(model_name))
            model_path = os.path.join(folder_path, "{}.pth".format(model_name))
            loaded_dict = torch.load(model_path, map_location=self.device)
            if model_name == 'encoder':
                self.epoch = loaded_dict.pop('epoch') + 1
                if isinstance(self.trainer, DDP):
                    self.trainer.module.epoch = self.epoch
                else:
                    self.trainer.epoch = self.epoch
            model.load_state_dict(loaded_dict)

        # loading optimizer state
        optimizer_path = os.path.join(folder_path, "adam.pth")
        if os.path.isfile(optimizer_path):
            print("Loading optimizer state...")
            optimizer_dict = torch.load(optimizer_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
            for state in self.model_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            print("No optimizer state found, initializing randomly.")




if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_folder", required=True, type=str,
                        help="Path to the folder containing saved model weights")
    args = parser.parse_args()

    # load options from the saved experiment configuration
    options_path = os.path.join(args.weights_folder, "opt.json")
    with open(options_path, "r") as f:
        opt = argparse.Namespace(**json.load(f))

    # initialize trainer and load saved model weights
    trainer = Trainer(opt)
    trainer.load_model()
    trainer.train()






