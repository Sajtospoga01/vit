
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import reduce
import torch
from torch import nn
from operator import mul
from typing import Callable, Optional, Tuple, Union
from src.dinov2.layers import Mlp, PatchEmbed,SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from src.dinov2.layers.attention import Attention


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class SpectralPatchEmbed(nn.Module):
    def __init__(self,
        img_size: Union[int, Tuple[int, int]] = 64,
        spectral_patch_size: Union[int, Tuple[int, int]] = 2,
        in_chans: int = 32,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
        ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        spectral_split = in_chans // spectral_patch_size
        local_crop_size = make_2tuple(img_size//2)

        self.global_crop_size = image_HW
        self.local_crop_size = local_crop_size
        self.spectral_patch_size = spectral_patch_size
        self.spectral_split = spectral_split

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        
        self.flatten_embedding = flatten_embedding


        self.proj_global = nn.Conv1d(image_HW[0] * image_HW[1], embed_dim, kernel_size=spectral_patch_size, stride=spectral_patch_size)
        self.proj_local = nn.Conv1d(local_crop_size[0] * local_crop_size[1], embed_dim, kernel_size=spectral_patch_size, stride=spectral_patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
       

        x = rearrange(x, 'b c h w -> b (h w) c')
        if H == self.local_crop_size[0] and W == self.local_crop_size[1]:
            x = self.proj_local(x)
        else:
            x = self.proj_global(x)

        x = x.transpose(1, 2)
        x = self.norm(x)
      
        return x
    

class FeatureFusion(nn.Module):
    def __init__(self,img_size,spatial_patches,spatial_local_patches,spectral_patches,register_tokens) -> None:
        super().__init__()

        spatial_size = spatial_patches**2+register_tokens+1
        spectral_size = spectral_patches+register_tokens+1
        spatial_local_size = spatial_local_patches**2+register_tokens+1
        
        self.spatial_size = spatial_size
        self.spectral_size = spectral_size
        self.spatial_local_size = spatial_local_size

        self.img_size = img_size
        self.fusion_spatial_global = nn.Conv1d(spatial_size + spectral_size, spatial_size, 1) # fuse features from 90 to 69 tokens
        self.fusion_spectral_global = nn.Conv1d(spatial_size + spectral_size, spectral_size, 1) # fuse features from 90 to 21 tokens

        self.fusion_spatial_local = nn.Conv1d(spatial_local_size + spectral_size, spatial_local_size, 1) # fuse features from 90 to 69 tokens
        self.fusion_spectral_local = nn.Conv1d(spatial_local_size + spectral_size, spectral_size, 1) # fuse features from 90 to 21 tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        B, N, F = x.shape
        
        if N == self.spatial_size + self.spectral_size :
            spatial = self.fusion_spatial_global(x)
            spectral = self.fusion_spectral_global(x)
        else:
            spatial = self.fusion_spatial_local(x)
            spectral = self.fusion_spectral_local(x)
        return spatial, spectral
    

class SpatialSpectralBlock(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        ffn_bias: bool = False,
        attn_class: nn.Module = Attention,
        drop_path: float = 0.,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        ffn_layer: Optional[Callable] = None,
        init_values: Optional[Tuple[float, float]] = None,
        img_size:int = 64,
        global_patches: int = 8,
        spectral_patches:int = 2,
        register_tokens: int = 1
        ) -> None:
        
        super().__init__()

        spatial_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            attn_class=attn_class,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
        )

        spectral_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            attn_class=attn_class,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
        )

        self.spatial_block = spatial_block
        self.spectral_block = spectral_block

        self.fusion = FeatureFusion(
            img_size=img_size,
            spatial_patches=global_patches,
            spatial_local_patches=global_patches//2,
            spectral_patches=spectral_patches,
            register_tokens=register_tokens
        )

    def forward(self, x_spatial: torch.Tensor,x_spectral) -> torch.Tensor:
        # debug_tensor("x_spatial", x_spatial)
        # debug_tensor("x_spectral", x_spectral)
        spatial, spectral = x_spatial,x_spectral
        spatial = self.spatial_block(spatial)
        spectral = self.spectral_block(spectral)
        if isinstance(x_spatial, list):
            spatial_out = []
            spectral_out = []

            for spa,spe in zip(spatial, spectral):
                out = torch.cat((spa, spe), dim=1)
         
                x_spatial,x_spectral = self.fusion(out)
                spatial_out.append(x_spatial + spa)
                spectral_out.append(x_spectral + spe)
            return spatial_out,spectral_out
        else:
            x = torch.cat((spatial, spectral), dim=1)
            x_spatial,x_spectral = self.fusion(x)
            return x_spatial + spatial,x_spectral+ spectral

def debug_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, is_contiguous={tensor.is_contiguous()}")

        
