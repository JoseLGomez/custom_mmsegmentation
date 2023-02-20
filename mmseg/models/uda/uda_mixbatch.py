import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
import kornia

from mmcv.parallel import MMDistributedDataParallel

from mmseg.core import add_prefix
from mmseg.models import BaseSegmentor, build_segmentor
from mmseg.models.builder import SEGMENTORS
from mmseg.utils.utils import downscale_label_ratio


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@SEGMENTORS.register_module()
class MixBatch(BaseSegmentor):

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()
        self.local_iter = 0
        #self.max_iters = cfg['max_iters']
        model_cfg = deepcopy(cfg['model'])
        self.model = build_segmentor(model_cfg)
        self.num_classes = 19
        self.enable_fdist = False
        self.batch_ratio = cfg['batch_ratio']
        self.apply_classmix = cfg['classmix']['apply']
        self.apply_classmix_rcs = cfg['classmix']['apply_rcs']
        self.classmix_rcs_apply_chance = cfg['classmix']['apply_rcs_chance']
        self.classmix_freq = cfg['classmix']['freq']
        self.rcs_classes = None
        self.rcs_classprob = None
        self.strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': cfg['classmix']['color_jitter_strength'],
            'color_jitter_p': cfg['classmix']['color_jitter_probability'],
            'blur': random.uniform(0, 1) if cfg['classmix']['blur'] else 0,
            'mean': None,  # assume same normalization
            'std': None
        }

        if 'imnet_feature_dist' in cfg.keys():
            self.enable_fdist = True
            self.fdist_lambda = cfg['imnet_feature_dist']['imnet_feature_dist_lambda']
            self.fdist_classes = cfg['imnet_feature_dist']['imnet_feature_dist_classes']
            self.fdist_scale_min_ratio = cfg['imnet_feature_dist']['imnet_feature_dist_scale_min_ratio']
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))


    def get_model(self):
        return get_module(self.model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        new_data_batch = dict()
        if self.batch_ratio[1] == 0:
            new_data_batch['img'] = data_batch['img'][:self.batch_ratio[0]]
            new_data_batch['gt_semantic_seg'] = data_batch['gt_semantic_seg'][:self.batch_ratio[0]]
            new_data_batch['img_metas'] = data_batch['img_metas'][:self.batch_ratio[0]]
        else:
            if self.apply_classmix and random.uniform(0, 1) > 1 - self.classmix_freq:
                dev = data_batch['img'].device
                means, stds = self.get_mean_std(data_batch['img_metas'], dev)
                self.strong_parameters['mean'] = means[0].unsqueeze(0)
                self.strong_parameters['std'] = stds[0].unsqueeze(0)
                # Apply mixing
                mixed_img, mixed_lbl = [None] * max(self.batch_ratio), [None] * max(self.batch_ratio)
                mix_masks = self.get_class_masks(data_batch['gt_semantic_seg'])

                for i in range(max(self.batch_ratio)):
                    self.strong_parameters['mix'] = mix_masks[i]
                    mixed_img[i], mixed_lbl[i] = self.strong_transform(
                        self.strong_parameters,
                        data=torch.stack((data_batch['img'][i], data_batch['pseudo_img'][i])),
                        target=torch.stack((data_batch['gt_semantic_seg'][i], data_batch['pseudolabel'][i])))
                if self.batch_ratio[0] == 0:
                    new_data_batch['img'] = torch.cat(mixed_img)
                    new_data_batch['gt_semantic_seg'] = torch.cat(mixed_lbl)
                    new_data_batch['img_metas'] = data_batch['pseudo_img_metas'][:self.batch_ratio[1]]
                else:
                    new_data_batch['img'] =  torch.cat((data_batch['img'][:self.batch_ratio[0]],
                                                        torch.cat(mixed_img)), 0)
                    new_data_batch['gt_semantic_seg'] = torch.cat((data_batch['gt_semantic_seg'][:self.batch_ratio[0]],
                                                                   torch.cat(mixed_lbl)), 0)
                    new_data_batch['img_metas'] = data_batch['img_metas'][:self.batch_ratio[0]] + \
                                          data_batch['pseudo_img_metas'][:self.batch_ratio[1]]
            else:
                if self.batch_ratio[0] == 0:
                    new_data_batch['img'] = data_batch['pseudo_img'][:self.batch_ratio[1]]
                    new_data_batch['gt_semantic_seg'] = data_batch['pseudolabel'][:self.batch_ratio[1]]
                    new_data_batch['img_metas'] = data_batch['pseudo_img_metas'][:self.batch_ratio[1]]
                else:
                    new_data_batch['img'] = torch.cat((data_batch['img'][:self.batch_ratio[0]],
                                                   data_batch['pseudo_img'][:self.batch_ratio[1]]), 0)
                    new_data_batch['gt_semantic_seg'] = torch.cat((data_batch['gt_semantic_seg'][:self.batch_ratio[0]],
                                                               data_batch['pseudolabel'][:self.batch_ratio[1]]), 0)
                    new_data_batch['img_metas'] = data_batch['img_metas'][:self.batch_ratio[0]] + \
                                              data_batch['pseudo_img_metas'][:self.batch_ratio[1]]

        #optimizer.zero_grad()
        if self.enable_fdist:
            losses, feat_loss, feat_log = self(**new_data_batch)
        else:
            losses = self(**new_data_batch)
        loss, log_vars = self._parse_losses(losses)
        #optimizer.step()
        if self.enable_fdist:
            loss.backward(retain_graph=self.enable_fdist)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward(retain_graph=self.enable_fdist)
        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))
        return outputs


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # ImageNet feature distance
        if self.enable_fdist:
            losses = self.model.forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
            src_feat = losses.pop('features')
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            return losses, feat_loss, feat_log
        else:
            losses = self.model.forward_train(img, img_metas, gt_semantic_seg)
            return losses
    

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def get_class_masks(self, labels):
        class_masks = []
        for label in labels:
            classes = torch.unique(labels)
            if 255 in classes:
                classes = classes[:-1]
            nclasses = classes.shape[0]
            if self.apply_classmix_rcs and self.rcs_classprob is not None \
                    and random.uniform(0, 1) > 1 - self.classmix_rcs_apply_chance:
                rcs_prob = self.rcs_classprob[classes.cpu().numpy()]
                rcs_prob = rcs_prob*(1/np.sum(rcs_prob))
                class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2),
                                                replace=False, p=rcs_prob)
            else:
                class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
            classes = classes[torch.Tensor(class_choice).long()]
            class_masks.append(self.generate_class_mask(label, classes).unsqueeze(0))
        return class_masks


    def generate_class_mask(self, label, classes):
        label, classes = torch.broadcast_tensors(label,
                                                 classes.unsqueeze(1).unsqueeze(2))
        class_mask = label.eq(classes).sum(0, keepdims=True)
        return class_mask

    def strong_transform(self, param, data=None, target=None):
        assert ((data is not None) or (target is not None))
        data, target = self.one_mix(mask=param['mix'], data=data, target=target)
        data, target = self.color_jitter(
            color_jitter=param['color_jitter'],
            s=param['color_jitter_s'],
            p=param['color_jitter_p'],
            mean=param['mean'],
            std=param['std'],
            data=data,
            target=target)
        data, target = self.gaussian_blur(blur=param['blur'], data=data, target=target)
        return data, target

    def get_mean_std(self, img_metas, dev):
        mean = [
            torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
            for i in range(len(img_metas))
        ]
        mean = torch.stack(mean).view(-1, 3, 1, 1)
        std = [
            torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
            for i in range(len(img_metas))
        ]
        std = torch.stack(std).view(-1, 3, 1, 1)
        return mean, std

    def one_mix(self, mask, data=None, target=None):
        if mask is None:
            return data, target
        if not (data is None):
            stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
            data = (stackedMask0 * data[0] +
                    (1 - stackedMask0) * data[1]).unsqueeze(0)
        if not (target is None):
            stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
            target = (stackedMask0 * target[0] +
                      (1 - stackedMask0) * target[1]).unsqueeze(0)
        return data, target
    
    def gaussian_blur(self, blur, data=None, target=None):
        if not (data is None):
            if data.shape[1] == 3:
                if blur > 0.5:
                    sigma = np.random.uniform(0.15, 1.15)
                    kernel_size_y = int(
                        np.floor(
                            np.ceil(0.1 * data.shape[2]) - 0.5 +
                            np.ceil(0.1 * data.shape[2]) % 2))
                    kernel_size_x = int(
                        np.floor(
                            np.ceil(0.1 * data.shape[3]) - 0.5 +
                            np.ceil(0.1 * data.shape[3]) % 2))
                    kernel_size = (kernel_size_y, kernel_size_x)
                    seq = nn.Sequential(
                        kornia.filters.GaussianBlur2d(
                            kernel_size=kernel_size, sigma=(sigma, sigma)))
                    data = seq(data)
        return data, target

    def color_jitter(self, color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
        # s is the strength of colorjitter
        if not (data is None):
            if data.shape[1] == 3:
                if color_jitter > p:
                    if isinstance(s, dict):
                        seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                    else:
                        seq = nn.Sequential(
                            kornia.augmentation.ColorJitter(
                                brightness=s, contrast=s, saturation=s, hue=s))
                    self.denorm_(data, mean, std)
                    data = seq(data)
                    self.renorm_(data, mean, std)
        return data, target

    def denorm(self, img, mean, std):
        return img.mul(std).add(mean) / 255.0


    def denorm_(self, img, mean, std):
        img.mul_(std).add_(mean).div_(255.0)


    def renorm_(self, img, mean, std):
        img.mul_(255.0).sub_(mean).div_(std)

