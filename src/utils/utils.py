import time
from urllib3.exceptions import IncompleteRead, ProtocolError
from botocore.exceptions import ResponseStreamingError
from concurrent.futures import ThreadPoolExecutor
import boto3
import os
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Timer:
    def __enter__(self):
        self.start_time = time.time()  # Store the current time when the timer starts

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self.start_time  # Calculate the elapsed time
        print(f"Execution time: {elapsed_time:.2f} seconds")



class MultipartDownloader:
    def __init__(self, bucket, key, part_size=25 * 1024 * 1024):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_PRIVATE_ACCESS_KEY"),
            config=boto3.session.Config(
                signature_version="s3v4",
                read_timeout=60 * 5,
                retries={"max_attempts": 10},
            ),
        )
        self.bucket = bucket
        self.key = key
        self.part_size = part_size

    def download_part(self, part_number, retries=3):
        start_byte = (part_number - 1) * self.part_size
        end_byte = start_byte + self.part_size - 1
        range_header = f"bytes={start_byte}-{end_byte}"

        for _ in range(retries):
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket, Key=self.key, Range=range_header
                )
                body = response["Body"].read()
                return start_byte, body
            except IncompleteRead as e:
                print(f"Retrying part {part_number} due to IncompleteRead: {e}")
            except ProtocolError as e:
                print(f"Retrying part {part_number} due to ProtocolError: {e}")
            except ResponseStreamingError as e:
                print(f"Retrying part {part_number} due to ResponseStreamingError: {e}")

            raise Exception(
                f"Failed to download part {part_number} after {retries} retries"
            )

    def download(self):
        # Get the size of the object
        response = self.s3_client.head_object(Bucket=self.bucket, Key=self.key)
        object_size = response["ContentLength"]

        # Calculate the number of parts
        num_parts = -(-object_size // self.part_size)  # Ceiling division

        # Download each part in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.download_part, part_number)
                for part_number in range(1, num_parts + 1)
            ]

        # Collect the parts into a bytearray
        data = bytearray(object_size)
        for future in futures:
            start_byte, part_data = future.result()
            data[start_byte: start_byte + len(part_data)] = part_data

        return data

class S3Connector:
    def __enter__(self):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_PRIVATE_ACCESS_KEY"),
            config=boto3.session.Config(
                signature_version="s3v4",
                read_timeout=60 * 2,
            ),
        )
        return self

    def __exit__(self, type, value, traceback):
        self.s3_client.close()


def normalize_image(image, mean, std,seed=0):
    """
    Normalize a multi-channel image using per-channel mean and standard deviation.

    Parameters:
        image (numpy.ndarray): The image to be normalized. Shape should be (height, width, channels).
        mean (list or numpy.ndarray): The mean values for each channel.
        std (list or numpy.ndarray): The standard deviation values for each channel.

    Returns:
        numpy.ndarray: The normalized image.
    """
    # Ensure mean and std are numpy arrays
    # Normalize each channel

    normalized_image = (image - mean) / std

    return normalized_image


class Normalize():
    def __init__(self, mean, std,device='cuda'):
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def __call__(self, x):
        normalized_batch = (x - self.mean) / self.std
        return normalized_batch
    



def resize_single_image(image, new_width, new_height):
    old_height, old_width, _ = image.shape

    # Create a set of indices for the new image
    x_indices = (np.arange(new_height) * (old_height / new_height)).astype(int)
    y_indices = (np.arange(new_width) * (old_width / new_width)).astype(int)

    # Use numpy's advanced indexing to pull out the correct pixels from the original image
    x_indices_mesh, y_indices_mesh = np.meshgrid(x_indices, y_indices, indexing='ij')
    # Index into the original image to get the resized image
    resized_image = image[x_indices_mesh, y_indices_mesh]

    return resized_image

def random_crop_and_resize(image, max_crop_percent, output_size, seed=0):
    """
    Perform a random crop and resize on an image using NumPy.

    Parameters:
    - image (numpy.ndarray): The input image array of shape (H, W, C).
    - max_crop_percent (float): The maximum percentage to crop from the original image.
    - output_size (tuple): The size of the output image (output_height, output_width).
    - seed (int, optional): The seed for the random number generator.

    Returns:
    - numpy.ndarray: The cropped and resized image.
    """

    state = np.random.RandomState(seed)


    height, width, _ = image.shape
    random_crop_percent = state.uniform(100.0 - max_crop_percent,100.0)
    crop_height = int(height * random_crop_percent / 100)
    crop_width = int(width * random_crop_percent / 100)

    y1 = np.random.randint(0, height - crop_height + 1)
    x1 = np.random.randint(0, width - crop_width + 1)
    cropped_image = image[y1:y1 + crop_height, x1:x1 + crop_width]
    if crop_height == height and crop_width == width:
        return cropped_image  # Skip resizing if no cropping was done
    new_height, new_width = output_size
    resized_image = resize_single_image(cropped_image, new_width, new_height)
    return resized_image



class RandomCropAndResize:
    def __init__(self, output_size, scale=(0.08, 1.0), seed=0):
        self.output_size = output_size
        self.scale = scale
        self.seed = seed

    def __call__(self, images):
 
        B, C, H, W = images.shape
        area = H * W

        # Generate random parameters for all images in the batch
        target_areas = torch.empty(B).uniform_(*self.scale) * area
        aspect_ratios = torch.empty(B).uniform_(3./4., 4./3.)

        ws = torch.round((target_areas * aspect_ratios) ** 0.5).int()
        hs = torch.round((target_areas / aspect_ratios) ** 0.5).int()

        # Correct sizes that are too large
        ws = torch.where(ws <= W, ws, W * torch.ones_like(ws))
        hs = torch.where(hs <= H, hs, H * torch.ones_like(hs))

        # Calculate random start positions for each image in the batch
        x1s = torch.zeros(B, dtype=torch.int32)
        y1s = torch.zeros(B, dtype=torch.int32)
        for i in range(B):
            x1s[i] = torch.randint(0, W - ws[i].item() + 1, (1,))
            y1s[i] = torch.randint(0, H - hs[i].item() + 1, (1,))

        # Perform cropping and resizing
        cropped_resized_images = torch.stack([
            F.interpolate(
                images[b:b+1, :, y1:y1+h, x1:x1+w],
                size=self.output_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            for b, (x1, y1, w, h) in enumerate(zip(x1s, y1s, ws, hs))
        ])

        return cropped_resized_images
    
class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image):
        image = F.adjust_brightness(image, torch.FloatTensor(1).uniform_(max(0, 1 - self.brightness), 1 + self.brightness).item())
        image = F.adjust_contrast(image, torch.FloatTensor(1).uniform_(max(0, 1 - self.contrast), 1 + self.contrast).item())
        image = F.adjust_saturation(image, torch.FloatTensor(1).uniform_(max(0, 1 - self.saturation), 1 + self.saturation).item())
        image = F.adjust_hue(image, torch.FloatTensor(1).uniform_(-self.hue, self.hue).item())
        return image
    
    def adjust_brightness(image, brightness_factor):
        """
        Adjust brightness of an image.

        Parameters:
        - image (torch.Tensor): Image tensor of shape (B, C, H, W)
        - brightness_factor (float): Factor to adjust the brightness. Should be >= 0.

        Returns:
        - torch.Tensor: Brightness adjusted image.
        """
        return image * brightness_factor
    
class RandomApply:
    def __init__(self, transform, p=0.5):
        """
        Apply the given transform randomly with a probability 'p'.

        Parameters:
        - transform (callable): The image transformation to apply.
        - p (float): The probability with which to apply the transformation.
        """
        self.transform = transform
        self.p = p

    def __call__(self, img):
        """
        Call method.

        Parameters:
        - img (PIL Image or Tensor): Image to be transformed.

        Returns:
        - Transformed image or original image.
        """
        if random.random() < self.p:
            return self.transform(img)
        return img
    
class CustomTransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class RandomFlip:
    def __init__(self, p_horizontal=0.5, p_vertical=0.5, seed=0):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
        torch.manual_seed(seed)

    def __call__(self, images):
        """
        Perform a random horizontal and/or vertical flip on a batch of images.

        Parameters:
        - images (torch.Tensor): Batch of images with shape (B, C, H, W).

        Returns:
        - torch.Tensor: Batch of flipped images.
        """
        B, C, H, W = images.shape

        # Random horizontal flip
        horizontal_flip = torch.rand(B) < self.p_horizontal
        images[horizontal_flip] = images[horizontal_flip].flip(-1)  # Flipping along the width axis

        # Random vertical flip
        vertical_flip = torch.rand(B) < self.p_vertical
        images[vertical_flip] = images[vertical_flip].flip(-2)  # Flipping along the height axis

        return images


class RandomSolarize():
    def __init__(self, threshold=128, p=0.2):
        """
        Initialize the RandomSolarize parameters.

        Parameters:
        - threshold (int): The threshold for solarization. Pixels below this value will be inverted.
        - p (float): The probability of applying the solarization.
        """
        super(RandomSolarize, self).__init__()
        self.threshold = threshold
        self.p = p

    def __call__(self, imgs):
        """
        Apply solarization randomly to a batch of images.

        Parameters:
        - imgs (torch.Tensor): The input batch of image tensors of shape (B, C, H, W).

        Returns:
        - torch.Tensor: The batch of solarized images.
        """
        for i in range(imgs.size(0)):  # Iterate through each image in the batch
            if torch.rand(1).item() < self.p:
                # Invert pixel values below the threshold for each image in the batch
                mask = imgs[i] < self.threshold
                imgs[i][mask] = 255 - imgs[i][mask]

        return imgs


class GaussianBlur():
    """
    Apply Gaussian Blur to an image with multiple bands using PyTorch.
    This module can be used with GPU acceleration.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0, kernel_size: int = 9):
        super(GaussianBlur, self).__init__()
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.kernel_size = kernel_size

    def __call__(self, images):
        """
        Apply Gaussian blur to a batch of images, each with potentially different sigma.

        Parameters:
        - images (torch.Tensor): Batch of images with shape (B, C, H, W).

        Returns:
        - torch.Tensor: Batch of blurred images.
        """
        B, C, H, W = images.shape
        device = images.device  # Get the device from the images tensor
        blurred_images = []

        for i in range(B):
            # Randomly select sigma for each image and ensure it's on the correct device
            sigma = (torch.rand(1, device=device) * (self.radius_max - self.radius_min) + self.radius_min).item()
            
            # Create Gaussian kernel (on the correct device)
            x = torch.arange(self.kernel_size, device=device).float() - self.kernel_size // 2
            kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            kernel_1d /= kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d.expand(C, 1, self.kernel_size, self.kernel_size)

            # Apply the kernel to each image in the batch
            padding = self.kernel_size // 2
            blurred_image = F.conv2d(images[i:i+1], kernel_2d, padding=padding, groups=C)
            blurred_images.append(blurred_image)

        # Stack the blurred images back into a batch
        return torch.cat(blurred_images, dim=0)
    


class MaybeToTensor:
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        
        return  torch.Tensor(pic)




def load_cfg():
    return OmegaConf.load("configs/default_eval.yaml")




class Logger:
    def __init__(self) -> None:
        self.logs = []
    
    def info(self,log):
        self.logs.append(log)
        print(log)
    
    def save(self, path):
        with open(path, "w") as f:
            for log in self.logs:
                f.write(log)
                f.write("\n")


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None,n_spectral_tokens=None, mask_generator=None,spectral_mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])
    

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    

    B = len(collated_global_crops)
    N = n_tokens
    N_spectral = n_spectral_tokens

    n_samples_masked = int(B * mask_probability)
    
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    spectral_upperbound = 0
    
    masks_list = []
    spectral_mask_list = []

    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        spectral_mask_list.append(torch.BoolTensor(spectral_mask_generator(int(N_spectral * random.uniform(prob_min, prob_max)))))

        upperbound += int(N * prob_max)
        spectral_upperbound += int(N_spectral * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
        spectral_mask_list.append(torch.BoolTensor(spectral_mask_generator(0)))

    random.shuffle(masks_list)
    random.shuffle(spectral_mask_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    collated_spectral_masks = torch.stack(spectral_mask_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    spectral_mask_indices_list = collated_spectral_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    spectral_masks_weight = (1 / collated_spectral_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_spectral_masks)[collated_spectral_masks]
    # spectral masks


    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        
        "upperbound": upperbound,
        "spectral_upperbound": spectral_upperbound,

        "collated_spectral_masks": collated_spectral_masks,
        "spectral_mask_indices_list": spectral_mask_indices_list,
        "spectral_masks_weight": spectral_masks_weight,
        "spectral_upperbound": spectral_upperbound,


        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "n_masked_spectral_patches": torch.full((1,), fill_value=spectral_mask_indices_list.shape[0], dtype=torch.long),
    }

class CollateDataAndCast:
    def __init__(self, mask_ratio_tuple, mask_probability, dtype, n_tokens=None,n_spectral_tokens=None, mask_generator=None,spectral_mask_generator=None):
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator
        self.spectral_mask_generator = spectral_mask_generator
        self.n_spectral_tokens = n_spectral_tokens

    def __call__(self, samples_list):
        return collate_data_and_cast(samples_list, self.mask_ratio_tuple, self.mask_probability, self.dtype, self.n_tokens,self.n_spectral_tokens, self.mask_generator,self.spectral_mask_generator)

