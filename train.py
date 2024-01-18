import random
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import io
import wandb
from keys import load_env
from src.vit_model.adapter import ModelAdapter, InputAdapter, OutputAdapter
from src.vit_model.mae import MaskedAutoencoderViT
from src.utils.data_loader import FlowGeneratorExperimental
from src.utils.data_loader_strategy import DataFactoryStrategy, BatchReaderStrategyProt
from src.utils.metrics import Loss, Accuracy, SSIMAccuracy
from src.utils.utils import MultipartDownloader, random_crop_and_resize, normalize_image
from src.utils.callbacks import ModelCheckpoint
from utilities.segmentation_utils.reading_strategies import RGBImageStrategy, HSImageStrategyMultiThread, \
    RasterImageStrategyMultiThread, BatchReaderStrategy
from utilities.segmentation_utils.constants import (
    FileType,
    ImageOrdering,
)
from utilities.segmentation_utils.ImagePreprocessor import PreprocessingQueue, PreFunction, random_flip_up_down, \
    random_flip_left_right

TRAINING_DATA_PATH = "/nfs/datasets/bchsi/pb_tr/"
VALIDATION_DATA_PATH = "/nfs/datasets/bchsi/pb_val/"

NUM_CLASSES = 25

mean_per_band = np.array([
    136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
    137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
    136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
    136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
    137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
    137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
    138.00128494, 138.17846624
])
std_per_band = np.array([
    33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
    33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
    34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
    33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
    33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
    33.27907802, 32.90732107
])
HPARAMS = {
    'mini_batch_size': 32,
    'batch_size': 256,
    'epoch': 1,
    'criterion': nn.MSELoss,
    'optimizer': torch.optim.Adam,
    'optimizer_params': {
        'lr': 0.0001,
    },

    # loader parameters
    'shuffle': True,
    'preprocess': True,

    # model paramters
    "model_parameters": {
        "patch_size": 8,
        # encoder specific params
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        # decoder specific params
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
    },

    "io_params": {
        "input_size": (64, 64),
        "bands": 32,
        "output_size": (64, 64),
        "num_classes": NUM_CLASSES,
    },
}


class Tokenizer(nn.Module):
    def __init__(self, patch_width, patch_height):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.tokenize = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        )

    def forward(self, x, mask=None):
        token = self.tokenize(x)
        if not mask is None:
            token = token[:, mask]
        return token


class Trainer:
    def __init__(self, model, optimizer, criterion, device, tokenizer):
        self.model = model
        self.model = self.model.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.loss_metric = Loss()
        self.ssim_accuracy_metric = Accuracy()
        self.val_loss_metric = Loss()
        self.val_ssim_accuracy_metric = Accuracy()
        self.tokenizer = tokenizer
        self.warmup = 10

    def pca_on_images_pytorch(self, images, n_components):
        """
        Perform PCA on a batch of images using PyTorch.

        Args:
        images (torch.Tensor): A batch of images, shape (batch, channels, height, width).
        n_components (int): Number of principal components to keep.

        Returns:
        torch.Tensor: Transformed images, shape (batch, n_components).
        """
        batch, channels, height, width = images.shape

        # Move the channel to the last dimension and flatten the images
        images = images.permute(0, 2, 3, 1).reshape(-1, channels)

        # Center the data (subtract mean of each feature)
        mean = torch.mean(images, dim=0, keepdim=True)
        centered_images = images - mean

        # Perform SVD
        U, S, V = torch.linalg.svd(centered_images, full_matrices=False)

        # Selecting the principal components
        components = V.t()[:n_components]

        # Project the data onto principal components
        transformed_images = torch.mm(centered_images, components.t())

        # Reshape and move the components dimension back to the second position
        transformed_images = transformed_images.view(batch, height, width, n_components)
        transformed_images = transformed_images.permute(0, 3, 1, 2)

        return transformed_images

    def pca_on_image(self, tensor, n_components=None):
        """
        Perform PCA on the channel dimension of an image tensor.

        Args:
        tensor (torch.Tensor): An image tensor of shape (C, H, W).
        n_components (int, optional): Number of principal components to return. If None, all components are returned.

        Returns:
        torch.Tensor: Transformed tensor with principal components on channel dimension.
        """
        # Reshaping the tensor to 2D (C, H*W)
        C, H, W = tensor.shape
        tensor_2d = tensor.view(C, -1)  # Reshaped tensor is now (C, H*W)

        # Centering the data (subtract mean of each feature)
        mean = torch.mean(tensor_2d, dim=1, keepdim=True)
        centered_tensor = tensor_2d - mean

        # Performing SVD
        U, S, V = torch.linalg.svd(centered_tensor, full_matrices=False)

        # Selecting the principal components
        components = V.t()[:n_components] if n_components is not None else V.t()

        # Projecting the data onto principal components
        transformed_tensor = torch.mm(centered_tensor, components.t())

        # Reshaping back to original image format, if needed
        if n_components is not None:
            return transformed_tensor.view(n_components, H, W)
        else:
            return transformed_tensor.view(C, H, W)

    def _train_step(self, X, y):
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        self.optimizer.zero_grad()

        output, mask, _, loss = self.model(X)

        output_cpu = output.cpu()
        pred_tokens = self.tokenizer(output_cpu)
        true_tokens = self.tokenizer(y.cpu())

        loss.backward()
        output_cpu = output_cpu.detach()
        self.optimizer.step()

        return loss, output_cpu, y.cpu()

    def _val_step(self, X_val, y_val):
        X_val = torch.Tensor(X_val).to(self.device)
        y_val = torch.Tensor(y_val).to(self.device)

        output, mask, attention, _ = self.model(X_val)

        loss = self.criterion(output, y_val)
        # if epoch >= self.warmup:
        output = self.pca_on_images_pytorch(output, n_components=3)
        y_val = self.pca_on_images_pytorch(y_val, n_components=3)
        X_val = y_val
        output_cpu = output.cpu()

        return output_cpu, loss, X_val.cpu(), y_val.cpu(), attention

    def _update_metrics(self, loss, output_cpu, y, is_train=True):
        y = torch.Tensor(y)
        if is_train:
            self.loss_metric.update(loss.item())
            self.ssim_accuracy_metric.update(output_cpu, y)
        else:
            self.val_loss_metric.update(loss.item())
            self.val_ssim_accuracy_metric.update(output_cpu, y)

    def _log_metrics(self, is_train=True, extra_metrics=None):
        metrics = {
            "loss": self.loss_metric.compute(),
            "accuracy": self.ssim_accuracy_metric.compute(),

        } if is_train else {
            "val_loss": self.val_loss_metric.compute(),
            "val_accuracy": self.val_ssim_accuracy_metric.compute(),

        }
        if extra_metrics is not None:
            metrics.update(extra_metrics)
        wandb.log(metrics)

    def validate(self, validation_loader, epoch):
        i = 0
        self.model.eval()
        with torch.no_grad():
            tqdm_instance_val = tqdm(validation_loader,
                                     bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]{desc}")
            for X_val, y_val in tqdm_instance_val:
                output_cpu, loss, X_val, y_val, attention = self._val_step(X_val, y_val)

                self._update_metrics(loss, output_cpu, y_val, is_train=False)

                tqdm_instance_val.set_description(
                    f"val_loss {self.val_loss_metric.compute():.4f}\t val_acc {self.val_ssim_accuracy_metric.compute():.4f}")
                wandb_log = {}
                if i % 50 == 0:
                    # Get the first image from input and prediction
                    input_img = torch.Tensor(X_val[0])
                    predicted_img = output_cpu.detach()[0]

                    # Average the channels to get a single-channel image
                    input_img_avg = self.rescale_for_visualization(input_img)
                    predicted_img_avg = self.rescale_for_visualization(predicted_img)

                    attention_cpu = attention[-1]
                    attention_cpu = attention_cpu.cpu()
                    attention_cpu = attention_cpu.mean(dim=0)

                    # Concatenate input and predicted images side by side
                    concatenated = np.hstack(
                        (input_img_avg.numpy(), predicted_img_avg.detach().numpy())
                    )
                    # Your provided loop to visualize attention

                    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
                    for head in attention_cpu:
                        head = head.detach().numpy()

                    head_mean = np.mean(np.array(attention_cpu), axis=0)

                    axs.imshow(head_mean)
                    axs.axis('off')
                    axs.set_title(f'Attention Map')

                    wandb_log["Attention Map"] = wandb.Image(fig)
                    concatenated = np.moveaxis(concatenated, 0, -1)
                    wandb_log["Input and Predicted Images"] = wandb.Image(
                        concatenated
                    )

                    # plt.close(fig)  # Close the figure to free up memory

                # # Log the images and attention maps to wandb
                self._log_metrics(is_train=False, extra_metrics=wandb_log)

                i += 1

    def train(self, epoch, train_loader, validation_loader=None, callbacks=None):
        torch.cuda.empty_cache()
        if "CL_WANDB_PROJECT" in os.environ:
            project = os.environ["CL_WANDB_PROJECT"]
            print(f"Loaded project {project}")
        else:
            project = "vit-dino"

        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config=HPARAMS,
        )
        

        for callback in callbacks:
            callback.add_model(self.model)
            callback.on_train_start()

        for epoch_idx in range(epoch):

            for callback in callbacks:
                callback.on_epoch_start()

            print(f"Epoch {epoch_idx + 1}/{epoch}:")

            tqdm_instance = tqdm(
                train_loader,
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]{desc}",
            )
            self.model.train()
            for X, y in tqdm_instance:

                loss, output_cpu, y = self._train_step(X, y)
                self._update_metrics(loss, output_cpu, y)
                self._log_metrics()
                tqdm_instance.set_description(
                    f"loss {self.loss_metric.compute():.4f}\t acc {self.ssim_accuracy_metric.compute():.4f}")
                for callback in callbacks:
                    callback.on_batch_end()

            if validation_loader is not None:
                self.validate(validation_loader, epoch_idx)
            logs = {
                "loss": self.loss_metric.compute(),
                "accuracy": self.ssim_accuracy_metric.compute(),
                "val_loss": self.val_loss_metric.compute(),
                "val_accuracy": self.val_ssim_accuracy_metric.compute(),
            }
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            train_loader.on_epoch_end()
            torch.cuda.empty_cache()

        for callback in callbacks:
            callback.on_train_end()

        wandb.finish()

    def rescale_for_visualization(self, image):
        """
        Rescales a PyTorch tensor image to the range [0, 1] for visualization.
        """
        min_val = torch.min(image)
        max_val = torch.max(image)
        return (image - min_val) / (max_val - min_val)

    def predict(self, image):
        self.model.eval()
        bogus_gpu = image.cuda()
        output, _ = self.model(bogus_gpu)
        output_cpu = output.cpu()
        # attention_cpu = attention
        return output_cpu  # attention_cpu


def main():

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    load_env()
    wandb.login()

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    in_queue = PreprocessingQueue(
        [
            PreFunction(random_flip_up_down),
            PreFunction(random_flip_left_right),
            PreFunction(random_crop_and_resize, max_crop_percent=40.0, output_size=HPARAMS['io_params']['input_size']),
            PreFunction(normalize_image, mean=mean_per_band, std=std_per_band),
        ],
    )

    val_queue = PreprocessingQueue(
        [
            PreFunction(normalize_image, mean=mean_per_band, std=std_per_band),
        ],
    )

    X_train = BatchReaderStrategy(
        os.path.join(TRAINING_DATA_PATH, "image"),
        image_size=HPARAMS['io_params']['input_size'],
    )

    X_val = BatchReaderStrategy(
        os.path.join(VALIDATION_DATA_PATH, "image"),
        image_size=HPARAMS['io_params']['input_size'],
    )

    batch_reader = DataFactoryStrategy(X_train)

    batch_reader_val = DataFactoryStrategy(X_val)

    reader_args = {
        "input_strategy": batch_reader,
        "output_strategy": batch_reader,
        "shuffle": HPARAMS["shuffle"],
        "preprocessing_enabled": HPARAMS["preprocess"],
        "channel_mask": [True for _ in range(32)],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
        "image_ordering": ImageOrdering.CHANNEL_FIRST,
        "type": [FileType.MULTICHANNEL, FileType.MULTICHANNEL],
        "preprocessing_queue_image": in_queue,
        "preprocessing_queue_mask": in_queue,
    }

    val_reader_args = {
        "input_strategy": batch_reader_val,
        "output_strategy": batch_reader_val,
        "shuffle": False,
        "preprocessing_enabled": True,
        "channel_mask": [True for _ in range(32)],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
        "image_ordering": ImageOrdering.CHANNEL_FIRST,
        "type": [FileType.MULTICHANNEL, FileType.MULTICHANNEL],
        "preprocessing_queue_image": val_queue,
        "preprocessing_queue_mask": val_queue,
    }

    reader = FlowGeneratorExperimental(**reader_args)
    val_reader = FlowGeneratorExperimental(**val_reader_args)
    reader.set_mini_batch_size(HPARAMS["mini_batch_size"])
    val_reader.set_mini_batch_size(HPARAMS["mini_batch_size"])

    AE = MaskedAutoencoderViT(
        img_size=HPARAMS['io_params']['input_size'],
        in_chans=HPARAMS['io_params']['bands'],
        **HPARAMS["model_parameters"],
    )

    total_params = sum(p.numel() for p in AE.parameters())
    print(f'{total_params:,} total parameters.')

    optimizer = HPARAMS["optimizer"](AE.parameters(), **HPARAMS["optimizer_params"])
    criterion = HPARAMS["criterion"]()

    callbacks = []

    tokenizer = Tokenizer(HPARAMS['model_parameters']['patch_size'], HPARAMS['model_parameters']['patch_size'])

    trainer = Trainer(AE, optimizer, criterion, device="cuda", tokenizer=tokenizer)
    epoch = HPARAMS["epoch"]
    checkpoint_callback = ModelCheckpoint("/nfs/best_hsi_mae_base.pth")
    callbacks.append(checkpoint_callback)
    print(len(reader))

    trainer.train(epoch, reader, callbacks=callbacks)

    torch.save(AE.state_dict(), "mae_hsi_pretrain_base.pth")


if __name__ == "__main":
    main()
