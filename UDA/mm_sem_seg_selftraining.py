import argparse
import copy
import sys
import random
import logging
import os
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import time
import math
import PIL.Image as Image
from PIL import ImageFont
from PIL import ImageDraw
import datetime
import matplotlib.pyplot as plt
import gc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmseg.apis import train_segmentor
from mmseg.utils import get_device, get_root_logger
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.pipelines import Compose

from cityscapesscripts.helpers.labels import trainId2label, labels

dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign'
                , 8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle'}

def selftraining_argument_parser():
    # Adds cotrainig arguments to the detectron2 base parser
    parser = argparse.ArgumentParser(description='Self-training en mmsegmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--unlabeled_dataset',
        dest='unlabeled_dataset',
        help='File with Data images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--init_weights',
        dest='init_weights',
        help='Initial weights to generate pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of selftraining rounds',
        default=10,
        type=int
    )
    parser.add_argument(
        '--max_unlabeled_samples',
        dest='max_unlabeled_samples',
        help='Number of maximum unlabeled samples',
        default=500,
        type=int
    )
    parser.add_argument(
        '--step_inc',
        dest='step_inc',
        help='Fix a image step to avoid consecutive images in secuences',
        default=1,
        type=int
    )
    parser.add_argument(
        '--continue_epoch',
        dest='continue_epoch',
        help='Continue self-training at the begining of the specified epoch',
        default=1,
        type=int
    )
    parser.add_argument(
        '--continue_training',
        dest='continue_training',
        help='Continue self-training from pseudolabels specified on epoch',
        default=None,
        type=int
    )
    parser.add_argument(
        '--no_progress',
        action='store_true'
    )
    parser.add_argument(
        '--scratch_training',
        help='Use pretrained model for training in each epoch',
        action='store_true'
    )
    parser.add_argument(
        '--best_model',
        help='Use the best model obtained during the epochs',
        action='store_true'
    )
    parser.add_argument(
        '--initial_score_A',
        dest='initial_score_A',
        help='Initial score to reach to propagate weights to the next epoch',
        default=0,
        type=float
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set a prefixed seed to random select the unlabeled data. Useful to replicate experiments',
        default=None,
        type=int
    )
    parser.add_argument(
        '--mask_file',
        dest='mask_file',
        help='Mask file to apply to pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--recompute_all_pseudolabels',
        help='Use source B statistics to order samples by less confidence',
        action='store_true'
    )
    parser.add_argument(
        '--use_param_weights',
        help='Force the weights on config in a continue_training',
        action='store_true'
    )
    parser.add_argument(
        '--only_pseudolabeling',
        help='Compute 1 cycle of pseudolabels only',
        action='store_true'
    )
    parser.add_argument(
        '--no_random',
        help='No random selection of pseudolabels',
        action='store_true'
    )
    parser.add_argument(
        '--prior_file',
        dest='prior_file',
        help='Class prior file from source dataset to apply to the pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_inference',
        dest='weights_inference',
        help='Initial weights to generate pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options
    return args


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def get_unlabeled_data(unlabeled_dataset, step_inc, seed, samples):
    with open(unlabeled_dataset,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    im_list.sort()
    init_indx = random.randrange(0, step_inc)
    indx_sampled = np.asarray(range(init_indx, len(im_list), step_inc), dtype=int)
    im_list = np.asarray(im_list)[indx_sampled]
    random.seed(seed)
    if samples > -1:
        im_list = random.sample(im_list.tolist(), min(len(im_list), samples))
    else:
        im_list = im_list.tolist()
    return im_list


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
    

def compute_mtp_thresholds(logger, pred_conf, pred_cls_num, tgt_portion, num_classes, min_conf, max_conf):
    threshold = []
    for i in range(num_classes):
        x = pred_conf[pred_cls_num == i]
        if len(x) == 0:
            threshold.append(0)
            continue        
        x = np.sort(x)
        logger.info("Class %s, pixels %.3f %%, mean %.2f, std %.2f" % (dict_classes[i], (len(x)/pred_conf.size)*100, np.mean(x), np.std(x)))
        if type(tgt_portion) == np.ndarray:
            threshold.append(x[np.int(np.round(len(x)*(1-tgt_portion[i])))])
        else:
            threshold.append(x[np.int(np.round(len(x)*(1-tgt_portion)))])
    threshold = np.array(threshold)
    threshold[threshold > max_conf] = max_conf
    threshold[threshold < min_conf] = min_conf
    return threshold


def apply_mpt(cfg, logger, outputs, num_classes, tgt_num, tgt_portion, void_label, mask_file=None, prior=None,
              prior_thres=0):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(logger, pred_conf, pred_cls_num, tgt_portion, num_classes, 
                                   cfg.pseudolabeling.min_conf, cfg.pseudolabeling.max_conf)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask_file is not None:
            mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
            label[mask] = void_Label
        prob = pred_conf[index]
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i, :, :].copy()
                prior_conf_mask[prior[i, :, :] >= prior_thres] = 1.0
                prior_conf_mask[prior[i, :, :] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux <= thres[i])*(label == i)] = np.nan
            else:
                label[(prob <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob <= thres[i])*(label == i)] = np.nan
        pseudolabels.append(label)
        # Compute image score using mean of the weighted confidence pixels values higher than the threshold cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        if void_label > num_classes:
            score = np.nanmean(prob[label != num_classes])
        else:
            score = np.nanmean(prob)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            if class_id < num_classes:
                aux_scores[class_id] = pixel_count[idx]
        if void_label > num_classes:
            # guarantee minimum of foreground in score system (5% on image)
            if (np.sum(aux_scores[:num_classes])/label.size)*100 < 5:
                aux_scores[-1] = 0
        scores_list.append(aux_scores)
    return np.asarray(pseudolabels), np.asarray(scores_list), np.asarray(pseudolabels_not_filtered), np.asarray(thres)


def inference_on_imlist(model, img_list, cfg):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    outputs = []
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    for img in img_list:
        # prepare data
        data = dict(img=img[0])
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        # forward the model
        with torch.no_grad():
            output = model.inference(data['img'][0],data['img_metas'][0], rescale=False)
            output = torch.squeeze(output)
            conf = torch.amax(output, 0).cpu().numpy()
            output_labels = torch.argmax(output, dim=0).to(torch.uint8).cpu().numpy()
            outputs.append([output_labels, conf])
    return outputs


def sorting_scores(scores, sorting_method, selftraining=False):
    if sorting_method == 'per_class':
        sorted_idx = np.lexsort((scores[:,-1],np.count_nonzero(scores[:,:-2], axis=1)))[::-1]
    elif sorting_method == 'per_void_pixels':
        # Sorting by number of void pixels (lower to higher)
        sorted_idx = np.argsort(scores[:,-2])
    elif sorting_method == 'confidence':
        # Sorting by confidence (lower to higher for cotraining)
        sorted_idx = np.argsort(scores[:,-1])
        if selftraining:
            # (higher to lower for selftraining)
            sorted_idx = sorted_idx[::-1][:len(scores)]
    else:
        #No sorting
        sorted_idx = np.arange(len(scores))
    return sorted_idx


def save_pseudolabels(images, pseudolabels, scores, pseudolabels_path, coloured_pseudolabels_path=None,
                      pseudolabels_not_filtered=None, coloured_pseudolabels_not_filtered_path=None, file_text=''):
    filenames_and_scores = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sfilenames_and_scores.txt' %
                                        (file_text))
    images_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sselected_images_path.txt' %
                              (file_text))
    psedolabels_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sselected_pseudolabels_path.txt' %
                                   (file_text))
    with open(filenames_and_scores,'w') as f:
        with open(images_txt,'w') as g:
            with open(psedolabels_txt,'w') as h:
                for idx, image in enumerate(images):
                    filename = image[0].split('/')[-1].split('.')[0] + '.png'
                    Image.fromarray(pseudolabels[idx]).save(os.path.join(pseudolabels_path,filename))
                    if coloured_pseudolabels_path is not None:
                        colour_label(pseudolabels[idx], os.path.join(coloured_pseudolabels_path,filename))
                    if pseudolabels_not_filtered is not None and coloured_pseudolabels_not_filtered_path is not None:
                        colour_label(pseudolabels_not_filtered[idx],
                                     os.path.join(coloured_pseudolabels_not_filtered_path, filename))
                    # Create txt with files names and scores
                    f.write('%s %s %s %s\n' % (filename, str(scores[idx][-1]), str(scores[idx][-2]),
                                               str(np.count_nonzero(scores[idx][:19]))))
                    g.write('%s\n' % (image[0]))
                    h.write('%s\n' % (os.path.join(pseudolabels_path,filename)))
    return images_txt, psedolabels_txt, filenames_and_scores


# Finds the best font size
def find_font_size(max_width, classes, font_file, max_font_size=100):

    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    # Find the maximum font size that all labels fit into the box width
    n_classes = len(classes)
    for c in range(n_classes):
        text = classes[c]
        for s in range(max_font_size, 1, -1):
            font = ImageFont.truetype(font_file, s)
            txt_size = draw.textsize(text, font=font)
            # print('c:{} s:{} txt_size:{}'.format(c, s, txt_size))
            if txt_size[0] <= max_width:
                max_font_size = s
                break

    # Find the maximum box height needed to fit the labels
    max_font_height = 1
    font = ImageFont.truetype(font_file, max_font_size)
    for c in range(n_classes):
        max_font_height = max(max_font_height,
                              draw.textsize(text, font=font)[1])

    return max_font_size, int(max_font_height)


# Draw class legend in an image
def draw_legend(w, color_map, classes, n_lines=3, txt_color=(255, 255, 255),
                font_file="UDA/fonts/Cicle_Gordita.ttf"):

    # Compute legend sizes
    n_classes = len(color_map)
    n_classes_per_line = int(math.ceil(float(n_classes) / n_lines))
    class_width = int(w/n_classes_per_line)
    font_size, class_height = find_font_size(class_width, classes, font_file)
    font = ImageFont.truetype(font_file, font_size)

    # Create PIL image
    img_pil = Image.new('RGB', (w, n_lines*class_height))
    draw = ImageDraw.Draw(img_pil)

    # Draw legend
    for i in range(n_classes):
        # Get color and label
        color = color_map[i]
        text = classes[i]

        # Compute current row and col
        row = int(i/n_classes_per_line)
        col = int(i % n_classes_per_line)

        # Draw box
        box_pos = [class_width*col, class_height*row,
                   class_width*(col+1), class_height*(row+1)]
        draw.rectangle(box_pos, fill=color, outline=None)

        # Draw text
        txt_size = draw.textsize(text, font=font)[0]
        txt_pos = [box_pos[0]+((box_pos[2]-box_pos[0])-txt_size)/2, box_pos[1]]
        draw.text(txt_pos, text, txt_color, font=font)

    return np.asarray(img_pil)


def colour_label(inference, filename):
    pred_colour = 255 * np.ones([inference.shape[0],inference.shape[1],3], dtype=np.uint8)
    for train_id, label in trainId2label.items():
        pred_colour[(inference == train_id),0] = label.color[0]
        pred_colour[(inference == train_id),1] = label.color[1]
        pred_colour[(inference == train_id),2] = label.color[2]
    color_map = []
    classes = []
    for i in range(len(trainId2label)):
        if i in trainId2label:
            color_map.append(trainId2label[i].color)
            classes.append(trainId2label[i].name)
    legend = draw_legend(inference.shape[1], color_map, classes, n_lines=2)
    pred_colour = np.concatenate((pred_colour, legend))
    Image.fromarray(pred_colour).save(filename)
    

def merge_txts_and_save(new_txt, txt1, txt2=None):
    if txt2 is not None:
        files = [txt1, txt2]
    else:
        files = [txt1]
    with open(new_txt, 'w') as f:
        for file in files:
            with open(file) as infile:
                for line in infile:
                    f.write(line)
    return new_txt


def update_best_score_txts_and_save(accum_scores_txt, accum_images_txt, accum_labels_txt,
                                    new_scores_txt, new_images_txt, new_labels_txt,
                                    save_img_txt, save_labels_txt, save_scores_txt,
                                    sorting_method):
    with open(accum_scores_txt,'r') as f:
        accum_scores = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_scores_txt,'r') as f:
        new_scores_txt = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_images_txt,'r') as f:
        accum_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_images_txt,'r') as f:
        new_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_labels_txt,'r') as f:
        accum_labels = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_labels_txt,'r') as f:
        new_labels = [line.rstrip().split(' ') for line in f.readlines()]
    ignore_list = []
    # Check for repeated images
    for idx, score in enumerate(new_scores_txt):
        for idx2, score2 in enumerate(accum_scores):
            if score[0] == score2[0]:
                # Depending of the sorting method we use scores or number of void pixel to update
                if sorting_method == 'per_class' or sorting_method == 'per_void_pixels':
                    check = score[2] < score2[2]
                else:
                    check = score[1] > score2[1]
                if check:
                    # If we found the same image with better score we updated values in all the acumulated lists
                    accum_scores[idx2][1] = score[1]
                    accum_scores[idx2][2] = score[2]
                    accum_scores[idx2][3] = score[3]
                    accum_labels[idx2] = new_labels[idx]
                # we store the index to do not add it again later
                ignore_list.append(idx)
                break
    # add new images into the accumulated ones
    for idx, score in enumerate(new_scores_txt):
        if idx not in ignore_list:
            accum_scores.append(score)
            accum_labels.append(new_labels[idx])
            accum_images.append(new_images[idx])
    # save each data in its respective txt
    new_img_dataset = open(save_img_txt,'w')
    new_labels_dataset = open(save_labels_txt,'w')
    new_scores_dataset = open(save_scores_txt,'w')
    for idx, _ in enumerate(accum_scores):
        new_img_dataset.write(accum_images[idx][0] + '\n')
        new_labels_dataset.write(accum_labels[idx][0] + '\n')
        new_scores_dataset.write(accum_scores[idx][0] + ' ' + accum_scores[idx][1] + ' ' + accum_scores[idx][2] + ' '
                                 + accum_scores[idx][3] + '\n')
    new_img_dataset.close()
    new_labels_dataset.close()
    new_scores_dataset.close()
    return save_img_txt, save_labels_txt, save_scores_txt


def get_checkpoint_info(model, checkpoint):
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        #model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        #model.PALETTE = dataset.PALETTE


def main():
    ## Initialization ##
    args = selftraining_argument_parser()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.work_dir
    work_root_dir = args.work_dir

    cfg.gpu_ids = [0]
    if args.weights_inference is not None:
        weights_pseudolabeling = args.weights_inference
        weights_train = args.weights_inference
    else:
        weights_pseudolabeling = cfg.model['pretrained']
        weights_train = cfg.model['pretrained']
    
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set a seed for the unlabeled data selection
    if args.seed is not None:
        cfg.seed = args.seed
    else:
        cfg.seed = random.randrange(0,2**32)
    continue_epoch = args.continue_epoch
    accumulated_selection_img = []
    pseudolabeling = cfg.pseudolabeling.mode
    accumulation_mode = cfg.pseudolabeling.accumulation
    num_selected = cfg.pseudolabeling.number
    tgt_portion = cfg.pseudolabeling.init_tgt_port
    if type(tgt_portion) == list:
        tgt_portion = np.asarray(tgt_portion)
        max_list_tgt = tgt_portion + cfg.pseudolabeling.max_tgt_port
    if args.prior_file is not None:
        source_priors = np.load(args.prior_file)
    else:
        source_priors = None
    prior_thres=0.1
    prior_relax=0.05
    void_label = cfg['model']['decode_head']['ignore_index']

    lr_config = copy.deepcopy(cfg['lr_config'])

    ### feature distance warmup from imagenet ###
    if 'imnet_feature_dist' in cfg['uda'] and  cfg['uda']['imnet_feature_dist']['warmup_step']:
        cfg.work_dir = os.path.join(work_root_dir,'fd_warmup')
        cfg_warmup = copy.deepcopy(cfg)
        cfg_warmup['runner']['max_iters'] = cfg['uda']['imnet_feature_dist']['warmup_iters']
        cfg_warmup['checkpoint_config']['interval'] = cfg['uda']['imnet_feature_dist']['warmup_iters']
        cfg_warmup['evaluation']['interval'] = cfg['uda']['imnet_feature_dist']['warmup_iters']/4
        cfg_warmup['model']['backbone']['norm_cfg']['type'] = 'BN'
        cfg_warmup['model']['decode_head']['norm_cfg']['type'] = 'BN'
        cfg_warmup['model']['auxiliary_head']['norm_cfg']['type'] = 'BN'
        cfg_warmup['uda']['batch_ratio'] = (sum(cfg['uda']['batch_ratio']),0)
        cfg_warmup['data']['train'] = cfg['data']['train']['source']
        model = build_segmentor(
                                cfg_warmup,
                                train_cfg=cfg_warmup.get('train_cfg'),
                                test_cfg=cfg_warmup.get('test_cfg'))
        logger.info("Feature distance warmup from imagenet %s" % (cfg_warmup.model['pretrained']))
        datasets = [build_dataset(cfg_warmup.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg_warmup.data.val)
            val_dataset.pipeline = cfg_warmup.data.val.pipeline
            datasets.append(build_dataset(val_dataset))
        train_segmentor(
                        model,
                        datasets,
                        cfg_warmup,
                        distributed=False,
                        validate=True,
                        timestamp=timestamp)
        weights_pseudolabeling = os.path.join(work_root_dir,'fd_warmup/latest.pth')
        weights_train = os.path.join(work_root_dir,'fd_warmup/latest.pth')

    ### start self-training ###
    if args.continue_training is not None:
        initial_epoch = args.continue_training
    else:
        initial_epoch = args.continue_epoch
    for epoch in range(initial_epoch,args.epochs+1):
        cfg.work_dir = os.path.join(work_root_dir,'model',str(epoch),'checkpoints')
        if continue_epoch > 1 and not args.only_pseudolabeling and not args.use_param_weights:
            weights_pseudolabeling = os.path.join(work_root_dir,'model',str(epoch-1),'checkpoints/latest.pth')
            if not args.scratch_training:
                weights_train = weights_pseudolabeling
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt,
                                        max_list_tgt,
                                        tgt_portion + cfg.pseudolabeling.tgt_port_step*(epoch-1))
            else:
                tgt_portion = min(tgt_portion + cfg.pseudolabeling.tgt_port_step*(epoch-1),
                                  cfg.pseudolabeling.max_tgt_port)
            prior_thres = max(prior_thres-(prior_relax*epoch-1), 0)
        if 'norm_cfg' in cfg['model']['backbone']:
            cfg['model']['backbone']['norm_cfg']['type'] = 'BN'
        if 'norm_cfg' in cfg['model']['decode_head']:
            cfg['model']['decode_head']['norm_cfg']['type'] = 'BN'
        if 'auxiliary_head' in cfg['model'] and 'norm_cfg' in cfg['model']['auxiliary_head']:
            cfg['model']['auxiliary_head']['norm_cfg']['type'] = 'BN'
        model = build_segmentor(
                                cfg,
                                train_cfg=cfg.get('train_cfg'),
                                test_cfg=cfg.get('test_cfg'))
        #model.init_weights()
        # experiments folders
        pseudolabels_path_model = os.path.join(work_root_dir,'model',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_model)
        coloured_pseudolabels_path_model = os.path.join(work_root_dir,'model',str(epoch),
                                                          'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_model)
        coloured_pseudolabels_not_filtered_path_model = os.path.join(work_root_dir,'model',str(epoch),
                                                                       'pseudolabeling/coloured_pseudolabels_not_filtered')
        create_folder(coloured_pseudolabels_not_filtered_path_model)
        dataset_path = os.path.join(work_root_dir,'model',str(epoch),'unlabeled_data_selected')
        create_folder(dataset_path)
        checkpoints_path = os.path.join(work_root_dir,'model',str(epoch),'checkpoints')
        create_folder(checkpoints_path)

        if args.continue_training is None:
            model.cuda()
            model.eval()
            logger.info("Starting training from cycle {}".format(epoch))
            logger.info("prepare unlabeled data")
            unlabeled_data = get_unlabeled_data(args.unlabeled_dataset, args.step_inc, cfg.seed, args.max_unlabeled_samples)
            logger.info("Compute inference on unlabeled data, model used from %s" % (weights_pseudolabeling))
            if epoch == 1:
                checkpoint = load_checkpoint(model.model, weights_pseudolabeling, map_location='cpu')
            else:
                checkpoint = load_checkpoint(model, weights_pseudolabeling, map_location='cpu')
            get_checkpoint_info(model, checkpoint)
            inference = inference_on_imlist(model, unlabeled_data, cfg)
            if pseudolabeling == 'mpt':
                start_time = time.perf_counter()
                pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh = \
                    apply_mpt(cfg, logger, inference, len(model.CLASSES), len(unlabeled_data),
                                  tgt_portion, void_label, args.mask_file, source_priors, prior_thres)
                total_time = time.perf_counter() - start_time
                logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))

            # Continue cotraining on the specified epoch
            if continue_epoch > 1:
                accumulated_selection_img = os.path.join(work_root_dir,'model',str(epoch-1),
                                                          'unlabeled_data_selected/dataset_img.txt')
                accumulated_selection_pseudo = os.path.join(work_root_dir,'model',str(epoch-1),
                                                             'unlabeled_data_selected/dataset_pseudolabels.txt')
                accumulated_scores = os.path.join(work_root_dir,'model',str(epoch-1),
                                                    'unlabeled_data_selected/filenames_and_scores.txt')
                continue_epoch = 1 #set to 1 to avoid reload the starting point in each loop when continue-training is set

            unlabeled_data = np.asarray(unlabeled_data)
            # Order pseudolabels by method selected on config file
            logger.info("Sorting mode: {}".format(cfg.pseudolabeling.sorting))
            sorted_idx = sorting_scores(scores_list, cfg.pseudolabeling.sorting, selftraining=True)
            sorted_scores_list = scores_list[sorted_idx]
            sorted_pseudolabels = pseudolabels[sorted_idx]
            sorted_unlabeled_data = unlabeled_data[sorted_idx]
            sorted_pseudolabels_not_filtered = pseudolabels_not_filtered[sorted_idx]
            # free memory
            del scores_list
            del pseudolabels
            del unlabeled_data
            del pseudolabels_not_filtered
            gc.collect()
            # save pseudolabels
            logger.info("Select candidates and Save on disk")
            # select candidates and save them to add them to the source data
            if len(sorted_unlabeled_data) > cfg.pseudolabeling.number:
                images_txt, psedolabels_txt, filenames_and_scores = \
                    save_pseudolabels(sorted_unlabeled_data[:num_selected], sorted_pseudolabels[:num_selected],
                                      sorted_scores_list[:num_selected], pseudolabels_path_model,
                                      coloured_pseudolabels_path_model, sorted_pseudolabels_not_filtered[:num_selected],
                                      coloured_pseudolabels_not_filtered_path_model)
            else:
                if pseudolabeling == 'gt_substraction':
                    images_txt, psedolabels_txt, filenames_and_scores = \
                        save_pseudolabels(sorted_unlabeled_data, sorted_pseudolabels, sorted_scores_list,
                                          pseudolabels_path_model, coloured_pseudolabels_path_model)
                else:
                    images_txt, psedolabels_txt, filenames_and_scores = \
                        save_pseudolabels(sorted_unlabeled_data, sorted_pseudolabels, sorted_scores_list,
                                          pseudolabels_path_model, coloured_pseudolabels_path_model,
                                          sorted_pseudolabels_not_filtered,
                                          coloured_pseudolabels_not_filtered_path_model)
        else:
            # continue_training specified
            accumulated_selection_img = os.path.join(work_root_dir,'model',str(args.continue_training),
                                                          'unlabeled_data_selected/dataset_img.txt')
            accumulated_selection_pseudo = os.path.join(work_root_dir,'model',str(args.continue_training),
                                                             'unlabeled_data_selected/dataset_pseudolabels.txt')
            accumulated_scores = os.path.join(work_root_dir,'model',str(args.continue_training),
                                                    'unlabeled_data_selected/filenames_and_scores.txt')
        if not args.only_pseudolabeling:
            if args.continue_training is None:
                # free memory
                del inference
                del sorted_unlabeled_data
                del sorted_pseudolabels
                del sorted_scores_list
                del sorted_pseudolabels_not_filtered
                gc.collect()
                # Compute data accumulation procedure
                logger.info("Compute data accumulation procedure selected: {}".format(accumulation_mode))
                if accumulation_mode is not None and len(accumulated_selection_img) > 0:
                    if accumulation_mode.lower() == 'all':
                        accumulated_selection_img = merge_txts_and_save(os.path.join(dataset_path,'dataset_img.txt'),
                                                                            accumulated_selection_img, images_txt)
                        accumulated_selection_pseudo = merge_txts_and_save(os.path.join(dataset_path,
                                                                                         'dataset_pseudolabels.txt'),
                                                                            accumulated_selection_pseudo,
                                                                            psedolabels_txt)
                        accumulated_scores = merge_txts_and_save(os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                                            accumulated_scores, filenames_and_scores)
                    if accumulation_mode.lower() == 'update_best_score':
                        accumulated_selection_img, accumulated_selection_pseudo, accumulated_scores = \
                            update_best_score_txts_and_save(
                                                        accumulated_scores, accumulated_selection_img,
                                                        accumulated_selection_pseudo,
                                                        filenames_and_scores, images_txt, psedolabels_txt,
                                                        os.path.join(dataset_path,'dataset_img.txt'),
                                                        os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                                        os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                        cfg.pseudolabeling.sorting)
                else:
                    # No accumulation, only training with new pseudolabels
                    accumulated_selection_img = merge_txts_and_save(os.path.join(dataset_path,'dataset_img.txt'),
                                                                            images_txt)
                    accumulated_selection_pseudo = merge_txts_and_save(os.path.join(dataset_path,
                                                                                     'dataset_pseudolabels.txt'),
                                                                            psedolabels_txt)
                    accumulated_scores = merge_txts_and_save(os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                                            filenames_and_scores)

             # Training step
            cfg.device = get_device()
            cfg['data']['train']['pseudolabels']['img_dir'] = accumulated_selection_img
            cfg['data']['train']['pseudolabels']['ann_dir'] = accumulated_selection_pseudo
            if 'uda' in cfg:
                cfg['data']['samples_per_gpu'] = max(cfg['uda']['batch_ratio'])

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.val.pipeline
                datasets.append(build_dataset(val_dataset))

            model = build_segmentor(
                                cfg,
                                train_cfg=cfg.get('train_cfg'),
                                test_cfg=cfg.get('test_cfg'))

            if 'ConcatDataset' not in str(type(datasets[0].source)) and datasets[0].source.rare_class_sampling:
                sorted_idx = np.argsort(datasets[0].source.rcs_classes)
                model.rcs_classes = np.asarray(datasets[0].source.rcs_classes)[sorted_idx]
                model.rcs_classprob = np.asarray(datasets[0].source.rcs_classprob)[sorted_idx]
                
            logger.info("Training starting model from %s" % (weights_train))
            if not args.scratch_training and epoch > 1:
                checkpoint = load_checkpoint(model, weights_train, map_location='cpu')
            else:
                checkpoint = load_checkpoint(model.model, weights_train, map_location='cpu')
            get_checkpoint_info(model, checkpoint)
            train_segmentor(
                            model,
                            datasets,
                            cfg,
                            distributed=False,
                            validate=True,
                            timestamp=timestamp)
            cfg['lr_config'] = lr_config
            if not args.no_progress:
                # The model for the next inference and training cycle is the last one obtained
                weights_pseudolabeling = os.path.join(cfg.work_dir,'latest.pth')
                if not args.scratch_training:
                    weights_train = weights_pseudolabeling
                    
            if epoch < 5 and args.recompute_all_pseudolabels:
                logger.info('Recompute pseudolabels')
                re_pseudolabels_path_model = os.path.join(work_root_dir,'model',str(epoch),
                                                            'pseudolabeling/recomputed_pseudolabels')
                create_folder(re_pseudolabels_path_model)
                re_coloured_pseudolabels_path = os.path.join(work_root_dir,'model',str(epoch),
                                                              'pseudolabeling/recomputed_coloured_pseudolabels')
                create_folder(re_coloured_pseudolabels_path)
                accumulated_img = get_data(accumulated_selection_img)
                accumulated_pseudo = get_data(accumulated_selection_pseudo)
                start_time = time.perf_counter()
                # Inference return a tuple of labels and confidences
                model.eval()
                inference = inference_on_imlist(model, accumulated_img, cfg)
                total_time = time.perf_counter() - start_time
                logger.info("Compute inference on unlabeled dataset A: {:.2f} s".format(total_time))
                logger.info("Pseudolabeling mode: {}".format(pseudolabeling))
                if pseudolabeling == 'mpt':
                    start_time = time.perf_counter()
                    pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh = \
                        apply_mpt(cfg, logger, inference, len(model.CLASSES), len(accumulated_pseudo), tgt_portion,
                                  void_label, args.mask_file, source_priors, prior_thres)
                    total_time = time.perf_counter() - start_time
                    logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
                else:
                    raise Exception('unknown pseudolabeling method defined')

                # select candidates and save them to add them to the source data
                images_txt, psedolabels_txt, filenames_and_scores = save_pseudolabels(accumulated_img,
                    pseudolabels, scores_list, re_pseudolabels_path_model, re_coloured_pseudolabels_path, 
                                                                                      file_text='recomp_')
                _, _, _ = update_best_score_txts_and_save(
                                                    accumulated_scores, accumulated_selection_img,
                                                    accumulated_selection_pseudo,
                                                    filenames_and_scores, images_txt, psedolabels_txt,
                                                    os.path.join(dataset_path,'dataset_img.txt'),
                                                    os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                    cfg.pseudolabeling.sorting)

                 # free memory
                del inference
                del scores_list
                del pseudolabels
                del accumulated_pseudo
                del accumulated_img
                gc.collect()

            # Update thesholdings
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt, max_list_tgt,
                                       tgt_portion + cfg.pseudolabeling.tgt_port_step)
            else:
                tgt_portion = min(tgt_portion + cfg.pseudolabeling.tgt_port_step, cfg.pseudolabeling.max_tgt_port)
            prior_thres = max(prior_thres-prior_relax, 0)
        cfg.seed = cfg.seed + epoch




if __name__ == "__main__":
    main()
