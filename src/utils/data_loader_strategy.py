from typing import Optional, Tuple, Any, Union, Sequence
import logging
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from src.utils.utils import S3Connector
import io
import os
from src.utils.utils import MultipartDownloader
from torchvision import transforms
import torch.functional as F
from src.utils.utils import Normalize, GaussianBlur, RandomCropAndResize, RandomFlip,CustomTransformCompose,RandomSolarize, ColorJitter,RandomApply




logger = logging.getLogger("dinov2")

class BatchReaderStrategyProt:
    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            package: Any = np,
            bands_enabled: Union[list[bool], object] = None,
    ) -> None:
        self.image_path = Path(image_path)
        meta_path = self.image_path.parent
        path = os.path.join(meta_path, "info.csv")
        print(f"Reading from path: {path}")
        with S3Connector() as s3:
            data = s3.s3_client.get_object(Bucket="paperspace-bucket", Key=path)
            body = data['Body'].read()
            df = pd.read_csv(io.BytesIO(body), index_col=0)

        # Read the info.csv file containing number of images and batch size the data was processed at
        # df = pd.read_csv(os.path.join(meta_path, "info.csv"), index_col=0)
        # Read the first row for n_image

        n_image = df.iloc[0][0]

        # Read the second row for batch_size
        batch_size = df.iloc[1][0]

        
        self.ex_batch_size = batch_size
        self.dataset_size = n_image
        # last batch of the dataset
        self.last_batch_idx = n_image // batch_size
        self.dataset_idxs = np.arange(self.last_batch_idx + 1)

        self.image_size = image_size
        self.package = package

        self.bands_enabled = bands_enabled
        self.inner_idx = np.arange(self.ex_batch_size)

    def resize_image_batch(self, image_batch, new_width, new_height):
        batch_size, old_height, old_width, _ = image_batch.shape

        # Create a set of indices for the new image
        x_indices = (np.arange(new_height) * (old_height / new_height)).astype(int)
        y_indices = (np.arange(new_width) * (old_width / new_width)).astype(int)

        # Use numpy's advanced indexing to pull out the correct pixels from the original image
        x_indices_mesh, y_indices_mesh = np.meshgrid(x_indices, y_indices, indexing='ij')

        # Repeat the indices arrays along the batch dimension
        x_indices_mesh = np.repeat(x_indices_mesh[np.newaxis, :, :], batch_size, axis=0)
        y_indices_mesh = np.repeat(y_indices_mesh[np.newaxis, :, :], batch_size, axis=0)

        # Index into the original image to get the resized images
        resized_images = image_batch[np.arange(batch_size)[:, np.newaxis, np.newaxis],
        x_indices_mesh, y_indices_mesh]

        return resized_images

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        idx = dataset_index // self.ex_batch_size
        idx = self.dataset_idxs[idx]

        data = MultipartDownloader("paperspace-bucket", f"{self.image_path}/batch_{idx}.npy").download()
        images = np.load(io.BytesIO(data))

        if self.bands_enabled is None:
            self.bands_enabled = [True] * images.shape[-1]
        images = images[:, :, :, self.bands_enabled]
        images = self.resize_image_batch(images, self.image_size[0], self.image_size[1])
        if idx == self.last_batch_idx and images.shape[0] != batch_size:
            return images[:batch_size, ...]
        images = images[self.inner_idx, ...]
        return images

    def get_dataset_size(self, mini_batch) -> int:
        return int(np.floor(self.dataset_size / float(mini_batch)))

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        remaining_idxs = self.dataset_idxs[:-1]
        # Shuffle the remaining batch indexes
        state.shuffle(remaining_idxs)
        # Append the last batch index back
        self.dataset_idxs = np.concatenate([remaining_idxs, [self.last_batch_idx]])
        state.shuffle(self.inner_idx)

class DataFactoryStrategy:
    def __init__(self, strategy):
        self.batch_reader = strategy
        self.cache = None
        self.call_count = 0
        self.prev_index = 0

    def read_batch(self, batch_size, dataset_index):
        if self.call_count == 0 and not self.cache is None:
            self.call_count += 1
            assert self.prev_index == dataset_index
        else:
            self.cache = None
            self.call_count = 0
            self.prev_index = dataset_index

        if self.cache is None:
            self.cache = self.batch_reader.read_batch(batch_size, dataset_index)

        return self.cache

    def get_dataset_size(self, mini_batch):
        return self.batch_reader.get_dataset_size(mini_batch)

    def get_image_size(self):
        return self.batch_reader.get_image_size()

    def shuffle_filenames(self, seed):
        self.batch_reader.shuffle_filenames(seed)


class DINOv2DataFactory():
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = CustomTransformCompose(
            [
                RandomCropAndResize(
                    global_crops_size, scale=global_crops_scale
                ),
                RandomFlip(p_horizontal=0.5, p_vertical=0.5),
            ]
        )

        self.geometric_augmentation_local = CustomTransformCompose(
            [
                RandomCropAndResize(
                    local_crops_size, scale=local_crops_scale
                ),
                RandomFlip(p_horizontal=0.5, p_vertical=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = CustomTransformCompose(
            [
                # RandomApply(
                #     ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                #     p=0.8,
                # ),
                # transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = CustomTransformCompose(
            [
                GaussianBlur(p=1.0),
            ]
        )
       

        global_transfo2_extra = CustomTransformCompose(
            [
                GaussianBlur(p=0.1),
                RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = CustomTransformCompose(
            [
                GaussianBlur(p=0.5)
            ]
        )
        

        # normalization
        self.normalize = CustomTransformCompose(
            [

                Normalize(mean=WHUOHS_DEFAULT_MEAN, std=WHUOHS_DEFAULT_STD),
            ]
        )

        self.global_transfo1 = CustomTransformCompose([color_jittering,global_transfo1_extra, self.normalize])
        self.global_transfo2 = CustomTransformCompose([color_jittering,global_transfo2_extra, self.normalize])
        self.local_transfo = CustomTransformCompose([color_jittering,local_transfo_extra, self.normalize])
        
    def __call__(self, images):
        B, C, H, W = images.shape  # Assuming images is a batch of shape (Batch, Channels, Height, Width)

        # Global crops
        im1_base = self.geometric_augmentation_global(images)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(images)
        global_crop_2 = self.global_transfo2(im2_base)

        # Local crops
        local_crops_batch = [
            self.local_transfo(self.geometric_augmentation_local(images)).squeeze(1) for _ in range(self.local_crops_number)
        ]

        # Organizing the output for each image in the batch
        output_list = []
        for b in range(B):
            image_output = {
                "global_crops": [global_crop_1[b].squeeze(), global_crop_2[b].squeeze()],
                "global_crops_teacher": [global_crop_1[b], global_crop_2[b]],
                "local_crops": [local_crop[b] for local_crop in local_crops_batch]
            }
            output_list.append(image_output)
        return output_list



# Use timm's names
WHUOHS_DEFAULT_MEAN = (
    136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
    137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
    136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
    136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
    137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
    137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
    138.00128494, 138.17846624
)
WHUOHS_DEFAULT_STD = (
    33.48886853, 33.22482796, 33.4670978,  33.53758141, 33.48675988, 33.33348355,
    33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
    34.80732573, 35.17761688, 35.1956623,  34.43121367, 33.76600779, 33.77061146,
    33.92844916, 34.0370747,  34.0285642,  33.87601205, 33.81035869, 33.66611756,
    33.74440912, 33.69755911, 33.69845938, 33.6707364,  33.62571536, 33.44615438,
    33.27907802, 32.90732107
)



# # This roughly matches torchvision's preset for classification training:
# #   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
# def make_classification_train_transform(
#     *,
#     crop_size: int = 64,
#     interpolation=transforms.InterpolationMode.BICUBIC,
#     hflip_prob: float = 0.5,
#     mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#     std: Sequence[float] = IMAGENET_DEFAULT_STD,
# ):
#     transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
#     if hflip_prob > 0.0:
#         transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
#     transforms_list.extend(
#         [
#             MaybeToTensor(),
#             make_normalize_transform(mean=mean, std=std),
#         ]
#     )
#     return transforms.Compose(transforms_list)