# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class CHESTNIH(MultiLabelDataset):

    CLASSES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis",
               "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax")

    def __init__(self, **kwargs):
        super(CHESTNIH, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from txt file.
        """
        data_infos = []


        img_ids = mmcv.list_from_file(self.ann_file)
        for img_id in img_ids:
            lineList = img_id.split(" ")
            filename, label = lineList[0], lineList[1:]
            # filenameFull = osp.join(self.data_prefix, filename)
            gt_label = np.array([int(item) for item in label])

            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)

        return data_infos
