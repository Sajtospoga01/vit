from mmseg.datasets import CustomDataset
import os
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.loading import LoadImageFromFile
import rasterio
import numpy as np
from mmseg.datasets.builder import DATASETS
import mmcv
from src.dinov2.eval.structures import BitMasks
import torch.nn.functional as F
from src.utils.data_loader_strategy import BatchReaderStrategyProt,S3Connector
import pandas as pd
from pathlib import Path
import io

@DATASETS.register_module()
class WHU_OHS(CustomDataset):
    CLASSES = ('background', 'paddy field', 'dry farm', 'woodland', 'shrubbery', 'Sparse woodland','Other forest land','High-covered grassland','Medium-covered grassland','Low-covered grassland', 'River canal', 'Lake', 'Reservoir pond', 'Beach land', 'Shoal', 'Urban built-up', 'Rural-settlement', 'Other construction land', 'Sand', 'Gobi', 'Saline-alkali soil', 'Marshland', 'Bare land', 'Bare rock', 'Ocean')
    PALETTE = [[0, 0, 0],[190, 210, 255],[0, 255, 197],[38, 115, 0],[163, 255, 115],[76, 230, 0],[85, 255, 0],[115, 115, 0],[168, 168, 0],[255, 255, 0],[115, 178, 255],[0, 92, 230],[0, 38, 115],[122, 142, 245],[0, 168, 132],[115, 0, 0],[255, 127, 127],[255, 190, 190],[255, 190, 232],[255, 0, 197],[230, 0, 169],[168, 0, 132],[115, 0, 76],[255, 115, 223],[161, 161, 161]]

    def __init__(self, **kwargs):
        super(WHU_OHS, self).__init__(
            img_suffix='.tif', seg_map_suffix='.tif', **kwargs)
        # assert os.path.exists(self.img_dir) and self.split is not None
        

@PIPELINES.register_module()
class WrapInList(object):
    """Wrap the results(dict) in list.

    This class can be used as a middle stage before ``Collect``.

    Args:
        keys (list[str]): Keys that need to be wrapped in list.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to wrap the results in list.

        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict with values wrapped in list.
        """
        
        for key in self.keys:
            results[key] = [results[key]]
        
        for key in results.keys():
            print(type(results[key]))
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class MyLoadImageFromFile(LoadImageFromFile):
    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk')):
        super().__init__(to_float32, color_type, file_client_args)

    def __call__(self, results):
   
        # Use rasterio or numpy to load the image

        if results.get('img_prefix') is not None:
            filename = os.path.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        with rasterio.open(filename) as src:
            image = src.read()
     
        # Transpose the image to HWC format
        image = np.transpose(image, (1, 2, 0))
        # Update results
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = image
        results['img_shape'] = image.shape
        results['ori_shape'] = image.shape
        results['pad_shape'] = image.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = None
        num_channels = image.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


@PIPELINES.register_module()
class LoadImageFromS3(LoadImageFromFile):
    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk')):
        super().__init__(to_float32, color_type, file_client_args)

    def __call__(self, results):
   
        # Use rasterio or numpy to load the image

        if results.get('img_prefix') is not None:
            filename = os.path.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        with rasterio.open(filename) as src:
            image = src.read()

        image = self.__get_image_with_info(filename)
     
        # Transpose the image to HWC format
        image = np.transpose(image, (1, 2, 0))
        # Update results
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = image
        results['img_shape'] = image.shape
        results['ori_shape'] = image.shape
        results['pad_shape'] = image.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = None
        num_channels = image.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
    
    # def __get_image_with_info(self, filename):



@PIPELINES.register_module()
class HSINormalize(object):
    """Normalize the HSI image with 32 bands.

    Args:
        mean (sequence): Mean values of 32 channels.
        std (sequence): Std values of 32 channels.
    """

    def __init__(self, mean, std):
        if len(mean) != 32 or len(std) != 32:
            raise ValueError("mean and std must have 32 elements each")
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img = results['img'].astype(np.float32)
   
        for i in range(img.shape[2]):  # Assuming img is in shape [H, W, C]
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        
        results['img'] = img
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, std={list(self.std)})'
        return repr_str
    

@PIPELINES.register_module()
class RepositionData(object):
    def __call__(self, results):
        results['img'] = results['img'].astype(np.float32)
        results['img'] = results['img'] / 10000.0
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class CastToFloat32(object):
    def __call__(self, results):
        results['img'] = results['img'].astype(np.float32)
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class ToMask(object):
    def __init__(self):
        self.ignore_label = 255  # Changed from 0 to 255
        self.num_classes = 25
        
    def __call__(self, results):
        sem_seg_gt = results['gt_semantic_seg']
     
        if sem_seg_gt is not None:
            classes = np.unique(sem_seg_gt)
        
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            results['gt_labels'] = torch.tensor(classes, dtype=torch.int64)
            
            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                results['gt_masks'] = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                results['gt_masks'] = masks.tensor

            sem_seg_gt_tensor = torch.from_numpy(sem_seg_gt).long()
            
            # Apply one-hot encoding
            # Ignore the ignore_label during one-hot encoding by setting it to zero
            sem_seg_gt_tensor[sem_seg_gt_tensor == self.ignore_label] = 0
            one_hot_targets = F.one_hot(sem_seg_gt_tensor, num_classes=self.num_classes + 1)  # +1 for the ignore label
            
            # Move the channels to the second dimension to match (N, C, H, W) format
            one_hot_targets = one_hot_targets.permute(2, 0, 1).float()
            
            # Remove the channel corresponding to the ignore label
            one_hot_targets = one_hot_targets[1:]  # Skip the first channel which is the ignore_label
            
            # Overwrite the semantic segmentation ground truth with the one-hot encoded tensor
            results['gt_semantic_seg'] = one_hot_targets.cpu().numpy()
            
          
            
        return results
   
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str