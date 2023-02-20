from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class UdaDataset(CustomDataset):

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, source, pseudolabels):
        self.source = source
        self.pseudolabels = pseudolabels
        #self.ignore_index = pseudolabels.ignore_index

    def __len__(self):
        return len(self.source) * len(self.pseudolabels)
    
    def __getitem__(self, idx):
        source = self.source[idx // len(self.pseudolabels)]
        pseudolabels = self.pseudolabels[idx % len(self.pseudolabels)]
        return {
                **source, 'pseudo_img_metas': pseudolabels['img_metas'],
                'pseudo_img': pseudolabels['img'], 'pseudolabel': pseudolabels['gt_semantic_seg']
            }