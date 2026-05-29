# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import time
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import NextScalePrediction, VQVAE
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, WandbLogger
from trainer import Trainer

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class NextScaleTrainer(Trainer):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, nsp_wo_ddp: NextScalePrediction, nsp: DDP,
        optimizer: AmpOptimizer, label_smooth: float, reweight_loss: bool = False,
        loss_reweight_type: str = 'equal',
    ):
        super(NextScaleTrainer, self).__init__(
            device, patch_nums, resos, vae_local, nsp_wo_ddp, nsp, optimizer, label_smooth, reweight_loss, loss_reweight_type
        )

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        L_resos = [0] * len(self.resos)
        acc_resos = [0] * len(self.resos)
        
        stt = time.time()
        
        training = self.transformer_wo_ddp.training
        self.transformer_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_ns_input(gt_idx_Bl)
            
            with torch.autocast('cuda', enabled=True, cache_enabled=True):
                logits_BLV = self.transformer_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            
            for si, (bg, ed) in enumerate(self.begin_ends):
                L_resos[si] += self.val_loss(logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)) * B
                acc_resos[si] += (logits_BLV.data[:, bg:ed].argmax(dim=-1) == gt_BL[:, bg:ed]).sum() * (100 / (ed - bg))
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.transformer_wo_ddp.train(training)
        
        stats = L_mean.new_tensor(L_resos + acc_resos + [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()[len(self.resos*2):]
        L_resos = stats.tolist()[:len(self.resos)]
        acc_resos = stats.tolist()[len(self.resos):len(self.resos*2)]
        return L_mean, L_tail, acc_mean, acc_tail, L_resos, acc_resos, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, wdb_lg: WandbLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], eval_labels: List[int], log_imgs_iters: int,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:

        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.transformer.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = self.quantize_local.idxBl_to_ns_input(gt_idx_Bl)
        
        with self.optimizer.amp_ctx:
            logits_BLV = self.transformer(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) 
            loss = loss.mul(self.loss_weight).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.optimizer.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to wandb
        if g_it == 0 or (g_it + 1) % 500 == 0:
            if dist.is_master():
                kw = {}
                tce = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
                tacc = (pred_BL == gt_BL).float().mean().item() * 100

                wdb_lg.update(head='Training Loss & Accuracy', **{'Total Loss': tce, 'Total Accuracy': tacc}, step=g_it)
    
                for si, (bg, ed) in enumerate(self.begin_ends):
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.patch_nums[si]}x{self.patch_nums[si]}'] = acc
                    kw[f'L_{self.patch_nums[si]}x{self.patch_nums[si]}'] = ce
                
                wdb_lg.update(head='Resolution Training Loss & Accuracy', **kw, step=g_it)

                if wdb_lg.initialized() and g_it == 0 or (g_it + 1) % log_imgs_iters == 0:
                    with torch.inference_mode():
                        labels = torch.tensor(eval_labels, device=dist.get_device(), dtype=torch.long)
                        imgs = self.transformer_wo_ddp.generate(len(eval_labels), labels, cfg=5.0, top_p=0.96, top_k=900, more_smooth=True)

                    wdb_lg.log_images('Visualization/Generation', imgs, step=g_it)

        return grad_norm, scale_log2
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('transformer_wo_ddp', 'vae_local', 'optimizer'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[NextScaleTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[NextScaleTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[NextScalePrediction.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)