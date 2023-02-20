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
from mmseg.apis import inference_segmentor, train_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.pipelines import Compose

from cityscapesscripts.helpers.labels import trainId2label, labels

dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign'
                , 8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle'}

def cotraining_argument_parser():
    parser = argparse.ArgumentParser(description='Co-training en mmsegmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--unlabeled_datasetA',
        dest='unlabeled_datasetA',
        help='File with Data A images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--unlabeled_datasetB',
        dest='unlabeled_datasetB',
        help='File with Data B images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of selftraining rounds',
        default=5,
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
        help='Continue co-training at the begining of the specified epoch',
        default=1,
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
        '--prior_file',
        dest='prior_file',
        help='Class prior file from source dataset to apply to the pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_inferenceA',
        dest='weights_inferenceA',
        help='Initial weights model A to generate pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_inferenceB',
        dest='weights_inferenceB',
        help='Initial weights model B to generate pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--min_pixels',
        dest='min_pixels',
        help='Minim number of pixels to filter a class on statistics',
        default=5000,
        type=int
    )
    parser.add_argument(
        '--ensembles',
        help='Generate pseudolabel with ensemble of the branches',
        action='store_true'
    )
    parser.add_argument(
        '--ensembles_subtraction',
        help='Mode subtraction on cotraining ensemble',
        action='store_true'
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
    parser.add_argument('--local_rank', type=int, default=0)
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


def apply_mpt(cfg, logger, outputs, num_classes, tgt_num, tgt_portion, mask_file=None, prior=None, prior_thres=0):
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
            label[mask] = 255
        prob = pred_conf[index]
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i, :, :].copy()
                prior_conf_mask[prior[i, :, :] >= prior_thres] = 1.0
                prior_conf_mask[prior[i, :, :] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux <= thres[i])*(label == i)] = 255  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux <= thres[i])*(label == i)] = np.nan
            else:
                label[(prob <= thres[i])*(label == i)] = 255  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob <= thres[i])*(label == i)] = np.nan
        pseudolabels.append(label)
        # Compute image score using mean of the weighted confidence pixels values higher than the threshold cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        score = np.nanmean(prob)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            if class_id < num_classes:
                aux_scores[class_id] = pixel_count[idx]
        scores_list.append(aux_scores)
    return pseudolabels, scores_list, pseudolabels_not_filtered, thres


def inference_on_imlist(weights_pseudolabeling, model, img_list, cfg):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    checkpoint = load_checkpoint(model, weights_pseudolabeling, map_location='cpu')
    prog_bar = mmcv.ProgressBar(len(img_list))
    get_checkpoint_info(model, checkpoint)
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
        prog_bar.update()
    return outputs


def sorting_scores(scoresA, scoresB, sorting_method, num_classes, info_inference_A, info_inference_B, cls_threshA,
                   cls_threshB, idx_to_remove=None):
    if sorting_method == 'per_class':
        sorted_idxA = np.lexsort((scoresA[:,-1],np.count_nonzero(scoresA[:,num_classes:-2], axis=1)))[::-1]
        sorted_idxB = np.lexsort((scoresB[:,-1],np.count_nonzero(scoresB[:,num_classes:-2], axis=1)))[::-1]
    elif sorting_method == 'per_void_pixels':
        # Sorting by number of void pixels (lower to higher)
        sorted_idxA = np.argsort(scoresA[:,-2])
        sorted_idxB = np.argsort(scoresB[:,-2])
    elif sorting_method == 'cotraining_confidence_score':
        # Sorting by score determined by the confidence difference of each class between branches
        conf_A = scoresA[:,0:num_classes]
        conf_B = scoresB[:,0:num_classes]
        conf_diffA = conf_B - conf_A
        conf_diffA[(conf_diffA < 0)] = 0
        conf_diffB = conf_A - conf_B
        conf_diffB[(conf_diffB < 0)] = 0
        new_scoresA = np.nanmean(conf_diffA * (1 - conf_A), axis=1)
        new_scoresB = np.nanmean(conf_diffB * (1 - conf_B), axis=1)
        sorted_idxA = np.lexsort((np.count_nonzero(scoresA[:,num_classes:-2], axis=1), new_scoresA))[::-1]
        sorted_idxB = np.lexsort((np.count_nonzero(scoresB[:,num_classes:-2], axis=1), new_scoresB))[::-1]
    elif sorting_method == 'by_confidence_on_class_demand':
        #Order by less confident classes
        sorted_idxA = compute_by_confidence_on_class_demand(cls_threshB, info_inference_A, scoresA)
        sorted_idxB = compute_by_confidence_on_class_demand(cls_threshA, info_inference_B, scoresB)
    elif sorting_method == 'by_confidence_difference_between_branches':
        #Order by less confident classes
        sorted_idxA = compute_by_confidence_on_class_demand(cls_threshB - cls_threshA, info_inference_A, scoresA)
        sorted_idxB = compute_by_confidence_on_class_demand(cls_threshA - cls_threshB, info_inference_B, scoresB)
    elif sorting_method == 'confidence':
        # Sorting by confidence lower to higher
        sorted_idxA = np.argsort(scoresA[:,-1])
        sorted_idxB = np.argsort(scoresB[:,-1])
        # higher to lower
        sorted_idxA = sorted_idxA[::-1][:len(scoresA)]
        sorted_idxB = sorted_idxB[::-1][:len(scoresB)]
    else:
        #No sorting
        sorted_idxA = np.arange(len(scoresA))
        sorted_idxB = np.arange(len(scoresB))
    # Delete idx not desired from filtering
    if idx_to_remove is not None and len(idx_to_remove) > 0:
        idx = np.where(np.in1d(sorted_idxA, idx_to_remove))[0]
        sorted_idxA = np.concatenate((np.delete(sorted_idxA, idx), sorted_idxA[idx]), axis=0)
        idx = np.where(np.in1d(sorted_idxB, idx_to_remove))[0]
        sorted_idxB = np.concatenate((np.delete(sorted_idxB, idx), sorted_idxB[idx]), axis=0)
    return sorted_idxA, sorted_idxB


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


def generate_pseudolabels(logger, cfg, model, weights_inference_branch, unlabeled_data,
                          tgt_portion, source_priors, prior_thres, mask_file):
    model.cuda()
    model.eval()
    start_time = time.perf_counter()
    # Inference return a tuple of labels and confidences
    inference = inference_on_imlist(weights_inference_branch, model, unlabeled_data, cfg)
    total_time = time.perf_counter() - start_time
    logger.info("Compute inference on unlabeled dataset: {:.2f} s".format(total_time))
    logger.info("Pseudolabeling mode: {}, Threshold: {}".format(cfg.pseudolabeling.mode, tgt_portion))
    if mask_file is not None:
        mask = Image.open(args.mask_file).convert('L')
        mask = mask.resize((cfg.pseudolabels_scale[0], cfg.pseudolabels_scale[1]))
        mask = np.asarray(mask, dtype=bool)
    else:
        mask = None
    if cfg.pseudolabeling.mode == 'mpt':
        start_time = time.perf_counter()
        pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh = \
                apply_mpt(cfg, logger, inference, len(model.CLASSES), len(unlabeled_data),
                              tgt_portion, mask_file, source_priors, prior_thres)
        total_time = time.perf_counter() - start_time
        logger.info("MPT on unlabeled dataset: {:.2f} s".format(total_time))
    else:
        raise Exception('unknown pseudolabeling method defined')

    return pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh
        

def recompute_pseudolabels(logger, cfg, model_name, model, weights_inference_branch, epoch, accumulated_selection_img,
                           accumulated_selection_pseudo, tgt_portion, source_priors, prior_thres,
                           accumulated_scores, dataset_path, work_root_dir, mask_file):
    logger.info("Recompute accumulated pseudolabels and update")
    re_pseudolabels_path_model = os.path.join(work_root_dir, model_name, str(epoch), 
                                              'pseudolabeling/recomputed_pseudolabels')
    create_folder(re_pseudolabels_path_model)
    re_coloured_pseudolabels_path = os.path.join(work_root_dir, model_name, str(epoch), 
                                                 'pseudolabeling/recomputed_coloured_pseudolabels')
    create_folder(re_coloured_pseudolabels_path)
    accumulated_img = get_data(accumulated_selection_img)
    accumulated_pseudo = get_data(accumulated_selection_pseudo)
    pseudolabels, scores_list, _, _, = generate_pseudolabels(logger, cfg, model, weights_inference_branch,
                                                             accumulated_pseudo, tgt_portion,
                                                             source_priors, prior_thres, mask_file)
    images_txt, psedolabels_txt, filenames_and_scores = save_pseudolabels(accumulated_img,
                    pseudolabels, scores_list, re_pseudolabels_path_model, re_coloured_pseudolabels_path, 
                                                                          file_text='recomp_')
    _, _, _ = update_best_score_txts_and_save(accumulated_scores, accumulated_selection_img, accumulated_selection_pseudo,
                                    filenames_and_scores, images_txt, psedolabels_txt,
                                    os.path.join(dataset_path,'dataset_img.txt'),
                                    os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                    os.path.join(dataset_path,'filenames_and_scores.txt'), cfg.pseudolabeling.sorting)


def compute_ensembles(logger, void_value, pseudolabels_A, pseudolabels_B, mode='+'):
    logger.info("Computing ensembling of pseudolabels")
    start_time = time.perf_counter()
    for idx, _ in enumerate(pseudolabels_A):
        aux_pseudoA = pseudolabels_A[idx].copy()
        aux_pseudoB = pseudolabels_B[idx].copy()
        # Common labels indexes
        idx_common = pseudolabels_A[idx] == pseudolabels_B[idx]
        # Not shared labels indexes
        idx_noncommon = pseudolabels_A[idx] != pseudolabels_B[idx]
        # From not shared labels indexes where A have a valid label and B is void
        idxA_novoid = (pseudolabels_B[idx] == void_value) * idx_noncommon
        # From not shared labels indexes where B have a valid label and A is void
        idxB_novoid = (pseudolabels_A[idx] == void_value) * idx_noncommon
        # Not shared labels indexes without void labels
        idx_noncommon = idx_noncommon * np.logical_not(np.logical_or(idxA_novoid, idxB_novoid))
        # Indexes from A common with B and indexes where A is void and B have a value
        idxA = np.logical_or(idx_common, idxB_novoid)
        # Indexes from B common with A and indexes where B is void and A have a value
        idxB = np.logical_or(idx_common, idxA_novoid)
        # Assign to aux A values from B where labels are common and A is void
        aux_pseudoA[idxA] = pseudolabels_B[idx][idxA]
        aux_pseudoB[idxB] = pseudolabels_A[idx][idxB]
        if mode == '+':
            # Assign to aux A values from A that differs with B
            aux_pseudoA[idx_noncommon] = pseudolabels_A[idx][idx_noncommon]
            aux_pseudoB[idx_noncommon] = pseudolabels_B[idx][idx_noncommon]
        else:
            # Assign not shared labels as void
            aux_pseudoA[idx_noncommon] = void_value
            aux_pseudoB[idx_noncommon] = void_value
        pseudolabels_A[idx] = aux_pseudoA
        pseudolabels_B[idx] = aux_pseudoB
    total_time = time.perf_counter() - start_time
    logger.info("Ensembles done in {:.2f} s".format(total_time))
    return pseudolabels_A, pseudolabels_B


def compute_statistics(logger, labels, dataset=None, inference_mode=False, min_pixels=5000):
    images = []
    categories = {}
    summary = {}
    info = {"images":images, "categories":categories, "summary":summary}
    if not inference_mode:
        dataset = get_data(labels)
    for train_id, label in trainId2label.items():
        if train_id >= 0:
            categories[train_id] = label[0]
            summary[train_id] = []
    for i in range(len(dataset)):
        if inference_mode:
            inference = labels[i]
        else:
            inference = np.asarray(Image.open(dataset[i][0]).convert('L'), dtype=np.uint8)
        classes = np.unique(inference, return_counts=True)
        image_dict = {  "id": i,
                    "file_name": dataset[i][0],
                    "classes": classes[0].tolist(),
                    "pixels": classes[1].tolist(),
                    "pixels_perc": (classes[1]/(inference.shape[1]*inference.shape[0])).tolist(),
                    "width": inference.shape[1],
                    "height": inference.shape[0]}
        images.append(image_dict)
        for idx, obj_cls in enumerate(classes[0]):
            if classes[1][idx] > min_pixels:
                summary[obj_cls].append(i)
        #print('%d/%d' % (i+1,len(labels)))
    logger.info("\n --- Summary --- \n")
    for idx, obj_cls in enumerate(categories):
        logger.info("Class %s: %d images, %.2f%%" % (categories[obj_cls], len(summary[obj_cls]), len(summary[obj_cls])*100/len(dataset)))
    logger.info("Total images: %d" % (len(dataset)))
    return info


def main():
    ## Initialization ##
    args = cotraining_argument_parser()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.work_dir
    work_root_dir = args.work_dir

    cfg.gpu_ids = [0]
    if args.weights_inferenceA is not None:
        weights_pseudolabelingA = args.weights_inferenceA
        weights_trainA = args.weights_inferenceA
    else:
        weights_pseudolabelingA = cfg.model['pretrained']
        weights_trainA = cfg.model['pretrained']
    if args.weights_inferenceB is not None:
        weights_pseudolabelingB = args.weights_inferenceB
        weights_trainB = args.weights_inferenceB
    else:
        weights_pseudolabelingB = cfg.model['pretrained']
        weights_trainB = cfg.model['pretrained']
    
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
    accumulated_selection_imgA = []
    accumulated_selection_imgB = []
    pseudolabeling = cfg.pseudolabeling.mode
    collaboration = cfg.pseudolabeling.collaboration
    accumulation_mode = cfg.pseudolabeling.accumulation
    num_selected = cfg.pseudolabeling.number
    tgt_portion = cfg.pseudolabeling.init_tgt_port
    if type(tgt_portion) == list:
        tgt_portion = np.asarray(tgt_portion)
        max_list_tgt = tgt_portion + cfg.pseudolabeling.max_tgt_port
    if cfg.pseudolabeling.init_tgt_port_B is not None:
        tgt_portion_B = cfg.pseudolabeling.init_tgt_port_B
        if type(tgt_portion_B) == list:
            tgt_portion_B = np.asarray(cfg.pseudolabeling.init_tgt_port_B)
            tgt_portion_B = np.asarray(tgt_portion_B)
            max_list_tgt_B = tgt_portion_B + cfg.pseudolabeling.max_tgt_port_B
    if args.prior_file is not None:
        source_priors = np.load(args.prior_file)
    else:
        source_priors = None
    prior_thres=0.1
    prior_relax=0.05
    void_value = cfg['model']['decode_head']['ignore_index']

    lr_config = copy.deepcopy(cfg['lr_config'])

    # debug #
    #print(cfg['data']['train'])
    #########
    for epoch in range(args.continue_epoch,args.epochs+1):
        cfg.work_dir = os.path.join(work_root_dir,'model_A',str(epoch),'checkpoints')
        if continue_epoch > 1 and not args.only_pseudolabeling and not args.use_param_weights:
            weights_pseudolabelingA = os.path.join(work_root_dir,'model_A',str(epoch-1),'checkpoints/latest.pth')
            weights_pseudolabelingB = os.path.join(work_root_dir,'model_B',str(epoch-1),'checkpoints/latest.pth')
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt,
                                        max_list_tgt, tgt_portion + cfg.pseudolabeling.tgt_port_step*epoch-1)
            else:
                tgt_portion = min(tgt_portion + cfg.pseudolabeling.tgt_port_step*epoch-1,
                                  cfg.pseudolabeling.max_tgt_port)
            if cfg.pseudolabeling.init_tgt_port_B is not None:
                if type(tgt_portion_B) == np.ndarray:
                    tgt_portion_B = np.where(tgt_portion_B >= max_list_tgt_B, max_list_tgt_B,
                                           tgt_portion_B + cfg.pseudolabeling.tgt_port_step_B*continue_epoch)
                else:
                    tgt_portion_B = min(tgt_portion_B + cfg.pseudolabeling.tgt_port_step_B*continue_epoch,
                                          cfg.pseudolabeling.max_tgt_port_B)
            else:
                tgt_portion_B = tgt_portion
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
        logger.info("Starting training from cycle {}".format(epoch))
        logger.info("prepare unlabeled data")
        unlabeled_dataA = get_unlabeled_data(args.unlabeled_datasetA, args.step_inc, cfg.seed,
                                             args.max_unlabeled_samples)
        if args.unlabeled_datasetB is not None:
            unlabeled_dataB = get_unlabeled_data(args.unlabeled_datasetB, args.step_inc, cfg.seed,
                                                 args.max_unlabeled_samples)
        else:
            unlabeled_dataB = unlabeled_dataA
        logger.info("Compute inference on unlabeled data, model used from %s" % (weights_pseudolabelingA))
        pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_threshA = \
                    generate_pseudolabels(logger, cfg, model, weights_pseudolabelingA,
                                          unlabeled_dataA, tgt_portion, source_priors, prior_thres, args.mask_file)
        logger.info("Compute inference on unlabeled data, model used from %s" % (weights_pseudolabelingB))
        pseudolabels_B, scores_listB, pseudolabels_B_not_filtered, cls_threshB = \
                    generate_pseudolabels(logger, cfg, model, weights_pseudolabelingB,
                                          unlabeled_dataB, tgt_portion_B, source_priors, prior_thres, args.mask_file)
        if args.ensembles and epoch > 1:
            if args.ensembles_subtraction:
                pseudolabels_A, pseudolabels_B = compute_ensembles(logger, void_value, pseudolabels_A,
                                                                   pseudolabels_B, '-')
            else:
                pseudolabels_A, pseudolabels_B = compute_ensembles(logger, void_value, pseudolabels_A, pseudolabels_B)

        logger.info("Computing pseudolabels statistics")
        start_time = time.perf_counter()
        info_inference_A = compute_statistics(logger, pseudolabels_A, unlabeled_dataA, True,
                                              min_pixels=args.min_pixels)
        total_time = time.perf_counter() - start_time
        logger.info("Statistics from branch A in {:.2f} s".format(total_time))
        start_time = time.perf_counter()
        info_inference_B = compute_statistics(logger, pseudolabels_B, unlabeled_dataB, True,
                                              min_pixels=args.min_pixels)
        total_time = time.perf_counter() - start_time
        logger.info("Statistics from branch B in {:.2f} s".format(total_time))

        # save pseudolabels
        pseudolabels_path_modelA = os.path.join(work_root_dir,'model_A',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_modelA)
        coloured_pseudolabels_path_modelA = os.path.join(work_root_dir,'model_A',str(epoch),
                                                          'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_modelA)
        coloured_pseudolabels_not_filtered_path_modelA = os.path.join(work_root_dir,'model_A',str(epoch),
                                                                       'pseudolabeling/coloured_pseudolabels_not_filtered')
        create_folder(coloured_pseudolabels_not_filtered_path_modelA)
        dataset_A_path = os.path.join(work_root_dir,'model_A',str(epoch),'unlabeled_data_selected')
        create_folder(dataset_A_path)
        checkpoints_A_path = os.path.join(work_root_dir,'model_A',str(epoch),'checkpoints')
        create_folder(checkpoints_A_path)
        pseudolabels_path_modelB = os.path.join(work_root_dir,'model_B',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_modelB)
        coloured_pseudolabels_path_modelB = os.path.join(work_root_dir,'model_B',str(epoch),
                                                          'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_modelB)
        coloured_pseudolabels_not_filtered_path_modelB = os.path.join(work_root_dir,'model_B',str(epoch),
                                                                       'pseudolabeling/coloured_pseudolabels_not_filtered')
        create_folder(coloured_pseudolabels_not_filtered_path_modelB)
        dataset_B_path = os.path.join(work_root_dir,'model_B',str(epoch),'unlabeled_data_selected')
        create_folder(dataset_B_path)
        checkpoints_B_path = os.path.join(work_root_dir,'model_B',str(epoch),'checkpoints')
        create_folder(checkpoints_B_path)

        # Continue cotraining on the specified epoch
        if continue_epoch > 1:
            accumulated_selection_imgA = os.path.join(work_root_dir,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_img.txt')
            accumulated_selection_pseudoA = os.path.join(work_root_dir,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_pseudolabels.txt')
            accumulated_scores_A = os.path.join(work_root_dir,'model_A',str(epoch-1),'unlabeled_data_selected/filenames_and_scores.txt')
            accumulated_selection_imgB = os.path.join(work_root_dir,'model_B',str(epoch-1),'unlabeled_data_selected/dataset_img.txt')
            accumulated_selection_pseudoB = os.path.join(work_root_dir,'model_B',str(epoch-1),'unlabeled_data_selected/dataset_pseudolabels.txt')
            accumulated_scores_B = os.path.join(work_root_dir,'model_B',str(epoch-1),'unlabeled_data_selected/filenames_and_scores.txt')
            continue_epoch = 0

        logger.info("Collaboration mode: {}".format(collaboration))
        scores_listA = np.asarray(scores_listA)
        pseudolabels_A = np.asarray(pseudolabels_A)
        unlabeled_datasetA = np.asarray(unlabeled_dataA)
        pseudolabels_A_not_filtered = np.asarray(pseudolabels_A_not_filtered)
        scores_listB = np.asarray(scores_listB)
        pseudolabels_B = np.asarray(pseudolabels_B)
        unlabeled_datasetB = np.asarray(unlabeled_dataB)
        pseudolabels_B_not_filtered = np.asarray(pseudolabels_B_not_filtered)
        # Order pseudolabels by method selected on config file
        if len(unlabeled_datasetA) < cfg.pseudolabeling.number:
            num_selected = len(unlabeled_datasetA)

        logger.info("Sorting mode: {}".format(cfg.pseudolabeling.sorting))
        start_time = time.perf_counter()
        if "self" in collaboration.lower(): #Self-training for each branch
            # Order pseudolabels by confidences (scores) higher to lower and select number defined to merge with
            # source data
            sorted_idxA, sorted_idxB = sorting_scores(scores_listA, scores_listB, cfg.pseudolabeling.sorting,
                                                      len(model.CLASSES), info_inference_A, info_inference_B,
                                                      cls_threshA, cls_threshB)

            sorted_scores_listA = scores_listA[sorted_idxA][:num_selected]
            sorted_pseudolabels_A = pseudolabels_A[sorted_idxA][:num_selected]
            sorted_unlabeled_datasetA = unlabeled_datasetA[sorted_idxA][:num_selected]
            sorted_pseudolabels_A_not_filtered = pseudolabels_A_not_filtered[sorted_idxA][:num_selected]

            sorted_scores_listB = scores_listB[sorted_idxB][:num_selected]
            sorted_pseudolabels_B = pseudolabels_B[sorted_idxB][:num_selected]
            sorted_unlabeled_datasetB = unlabeled_datasetB[sorted_idxB][:num_selected]
            sorted_pseudolabels_B_not_filtered = pseudolabels_B_not_filtered[sorted_idxB][:num_selected]

        if "cotraining" in collaboration.lower():
            # Order pseudolabels by confidence lower to higher and asign the less n confident to the other model
            sorted_idxA, sorted_idxB = sorting_scores(scores_listA, scores_listB, cfg.pseudolabeling.sorting, len(model.CLASSES), info_inference_A, info_inference_B, cls_threshA, cls_threshB)
            if "self" in collaboration.lower():
                sorted_scores_listA = np.concatenate((sorted_scores_listA[:int(num_selected/2)], scores_listB[sorted_idxB][:int(num_selected/2)]), axis=0)
                sorted_pseudolabels_A = np.concatenate((sorted_pseudolabels_A[:int(num_selected/2)], pseudolabels_B[sorted_idxB][:int(num_selected/2)]), axis=0)
                sorted_unlabeled_datasetA = np.concatenate((sorted_unlabeled_datasetA[:int(num_selected/2)], unlabeled_datasetA[sorted_idxB][:int(num_selected/2)]), axis=0)
                sorted_pseudolabels_A_not_filtered = np.concatenate((sorted_pseudolabels_A_not_filtered[:int(num_selected/2)], pseudolabels_B_not_filtered[sorted_idxB][:int(num_selected/2)]), axis=0)

                sorted_scores_listB = np.concatenate((sorted_scores_listB[:int(num_selected/2)], scores_listA[sorted_idxA][:int(num_selected/2)]), axis=0)
                sorted_pseudolabels_B = np.concatenate((sorted_pseudolabels_B[:int(num_selected/2)], pseudolabels_A[sorted_idxA][:int(num_selected/2)]), axis=0)
                sorted_unlabeled_datasetB = np.concatenate((sorted_unlabeled_datasetB[:int(num_selected/2)], unlabeled_datasetB[sorted_idxA][:int(num_selected/2)]), axis=0)
                sorted_pseudolabels_B_not_filtered = np.concatenate((sorted_pseudolabels_B_not_filtered[:int(num_selected/2)], pseudolabels_A_not_filtered[sorted_idxA][:int(num_selected/2)]), axis=0)
            else:
                sorted_scores_listA = scores_listB[sorted_idxB][:num_selected]
                sorted_pseudolabels_A = pseudolabels_B[sorted_idxB][:num_selected]
                sorted_unlabeled_datasetA = unlabeled_datasetA[sorted_idxB][:num_selected]
                sorted_pseudolabels_A_not_filtered = pseudolabels_B_not_filtered[sorted_idxB][:num_selected]

                sorted_scores_listB = scores_listA[sorted_idxA][:num_selected]
                sorted_pseudolabels_B = pseudolabels_A[sorted_idxA][:num_selected]
                sorted_unlabeled_datasetB = unlabeled_datasetB[sorted_idxA][:num_selected]
                sorted_pseudolabels_B_not_filtered = pseudolabels_A_not_filtered[sorted_idxA][:num_selected]

        if not "self" in collaboration.lower() and not "cotraining" in collaboration.lower():
            raise Exception('unknown collaboration of models defined')

        total_time = time.perf_counter() - start_time
        logger.info("Sorting done in {:.2f} s".format(total_time))

        # free memory
        del scores_listA
        del pseudolabels_A
        del unlabeled_datasetA
        del pseudolabels_A_not_filtered
        del scores_listB
        del pseudolabels_B
        del unlabeled_datasetB
        del pseudolabels_B_not_filtered
        gc.collect()

        # select candidates and save them to add them to the source data
        images_txt_A, psedolabels_txt_A, filenames_and_scoresA = save_pseudolabels(sorted_unlabeled_datasetA, sorted_pseudolabels_A, sorted_scores_listA, pseudolabels_path_modelA,
            coloured_pseudolabels_path_modelA, sorted_pseudolabels_A_not_filtered, coloured_pseudolabels_not_filtered_path_modelA)
        images_txt_B, psedolabels_txt_B, filenames_and_scoresB = save_pseudolabels(sorted_unlabeled_datasetB, sorted_pseudolabels_B, sorted_scores_listB, pseudolabels_path_modelB,
            coloured_pseudolabels_path_modelB, sorted_pseudolabels_B_not_filtered, coloured_pseudolabels_not_filtered_path_modelB)

        # free memory
        del sorted_unlabeled_datasetA
        del sorted_pseudolabels_A
        del sorted_scores_listA
        del sorted_pseudolabels_A_not_filtered
        del sorted_unlabeled_datasetB
        del sorted_pseudolabels_B
        del sorted_scores_listB
        del sorted_pseudolabels_B_not_filtered
        gc.collect()

        if not args.only_pseudolabeling:
            # Compute data accumulation procedure
            logger.info("Acumulation mode: {}".format(accumulation_mode.lower()))
            start_time = time.perf_counter()
            if accumulation_mode is not None and len(accumulated_selection_imgA) > 0:
                if accumulation_mode.lower() == 'all':
                    accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                        accumulated_selection_imgA, images_txt_A)
                    accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                        accumulated_selection_pseudoA, psedolabels_txt_A)
                    accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                        accumulated_scores_A, filenames_and_scoresA)
                    accumulated_selection_imgB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_img.txt'),
                                                                        accumulated_selection_imgB, images_txt_B)
                    accumulated_selection_pseudoB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                                        accumulated_selection_pseudoB, psedolabels_txt_B)
                    accumulated_scores_B = merge_txts_and_save(os.path.join(dataset_B_path,'filenames_and_scores.txt'),
                                                                        accumulated_scores_B, filenames_and_scoresB)
                elif accumulation_mode.lower() == 'update_best_score':
                    accumulated_selection_imgA, accumulated_selection_pseudoA, accumulated_scores_A = update_best_score_txts_and_save(
                                                    accumulated_scores_A, accumulated_selection_imgA, accumulated_selection_pseudoA,
                                                    filenames_and_scoresA, images_txt_A, psedolabels_txt_A,
                                                    os.path.join(dataset_A_path,'dataset_img.txt'),
                                                    os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_A_path,'filenames_and_scores.txt'), cfg.pseudolabeling.sorting)
                    accumulated_selection_imgB, accumulated_selection_pseudoB, accumulated_scores_B = update_best_score_txts_and_save(
                                                    accumulated_scores_B, accumulated_selection_imgB, accumulated_selection_pseudoB,
                                                    filenames_and_scoresB, images_txt_B, psedolabels_txt_B,
                                                    os.path.join(dataset_B_path,'dataset_img.txt'),
                                                    os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_B_path,'filenames_and_scores.txt'), cfg.pseudolabeling.sorting)
            else:
                #No accumulation, only training with new pseudolabels
                accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                        images_txt_A)
                accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                        psedolabels_txt_A)
                accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                        filenames_and_scoresA)
                accumulated_selection_imgB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_img.txt'),
                                                                        images_txt_B)
                accumulated_selection_pseudoB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                                        psedolabels_txt_B)
                accumulated_scores_B = merge_txts_and_save(os.path.join(dataset_B_path,'filenames_and_scores.txt'),
                                                                        filenames_and_scoresB)

            # Train model A

            cfg['data']['train']['pseudolabels']['img_dir'] = accumulated_selection_imgA
            cfg['data']['train']['pseudolabels']['ann_dir'] = accumulated_selection_pseudoA
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

            logger.info("Training starting model from %s" % (weights_trainA))
            checkpoint = load_checkpoint(model, weights_trainA, map_location='cpu')
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
                weights_pseudolabelingA = os.path.join(cfg.work_dir,'latest.pth')
                if not args.scratch_training:
                    weights_trainA = weights_pseudolabelingA

            # Train model B
            cfg.work_dir = os.path.join(work_root_dir,'model_B',str(epoch),'checkpoints')
            cfg['data']['train']['pseudolabels']['img_dir'] = accumulated_selection_imgB
            cfg['data']['train']['pseudolabels']['ann_dir'] = accumulated_selection_pseudoB
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

            logger.info("Training starting model from %s" % (weights_trainB))
            checkpoint = load_checkpoint(model, weights_trainB, map_location='cpu')
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
                weights_pseudolabelingB = os.path.join(cfg.work_dir,'latest.pth')
                if not args.scratch_training:
                    weights_trainB = weights_pseudolabelingB
                    
            if epoch < 5 and args.recompute_all_pseudolabels:
                logger.info("Recompute accumulated pseudolabels and update branch A")
                recompute_pseudolabels(logger, cfg, 'model_A', model, weights_pseudolabelingA, epoch, accumulated_selection_imgA,
                                       accumulated_selection_pseudoA, tgt_portion, source_priors, prior_thres,
                                       accumulated_scores_A, dataset_A_path, work_root_dir, args.mask_file)

                logger.info("Recompute accumulated pseudolabels and update branch B")
                recompute_pseudolabels(logger, cfg, 'model_B', model, weights_pseudolabelingB, epoch, accumulated_selection_imgB,
                                       accumulated_selection_pseudoB, tgt_portion_B, source_priors, prior_thres,
                                       accumulated_scores_B, dataset_B_path, work_root_dir, args.mask_file)

             # Update thesholdings
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt, max_list_tgt,
                                       tgt_portion + cfg.pseudolabeling.tgt_port_step)
            else:
                tgt_portion = min(tgt_portion + cfg.pseudolabeling.tgt_port_step, cfg.pseudolabeling.max_tgt_port)
            if cfg.pseudolabeling.init_tgt_port_B is not None:
                if type(tgt_portion_B) == np.ndarray:
                    tgt_portion_B = np.where(tgt_portion_B >= max_list_tgt_B, max_list_tgt_B,
                                       tgt_portion_B + cfg.pseudolabeling.tgt_port_step_B)
                else:
                    tgt_portion_B = min(tgt_portion_B + cfg.pseudolabeling.tgt_port_step_B,
                                      cfg.pseudolabeling.max_tgt_port_B)
            else:
                tgt_portion_B = tgt_portion
            prior_thres = max(prior_thres-prior_relax, 0)
        cfg.seed = cfg.seed + epoch

    '''if args.ensembles:
        ensembles_folder = os.path.join(work_root_dir,'final_ensemble')
        create_folder(ensembles_folder)
        modelA = build_model(cfg)
        modelB = build_model(cfg)    
        dataset_name = 'final_ensemble'
        inference_list = get_data(cfg.DATASETS.TEST_IMG_TXT)
        built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, dataset_name)
        if args.mpt_ensemble:
            if args.no_training:
                thresA = np.asarray(cfg.PSEUDOLABELING.INIT_TGT_PORT)
                thersB = np.asarray(cfg.PSEUDOLABELING.INIT_TGT_PORT_B)
                cls_thres = np.where(thresA <= thersB, thresA,
                                       thersB)
            else:
                cls_thres = min(cls_threshA, cls_threshB)
            print(cls_thres)
            ensemble_on_imlist_and_save(cfg, modelA, modelB, weights_inference_branchA, weights_inference_branchB, dataset_name, inference_list, ensembles_folder, evaluation=True, mask_file=args.mask_file, thres=cls_thres)
        else:
            ensemble_on_imlist_and_save(cfg, modelA, modelB, weights_inference_branchA, weights_inference_branchB, dataset_name, inference_list, ensembles_folder, evaluation=True)
    '''





if __name__ == "__main__":
    main()
