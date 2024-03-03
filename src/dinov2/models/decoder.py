import torch
from torch import nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from src.dinov2.layers import Mlp, PatchEmbed,SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from src.vit_model.custom_layers import get_2d_sincos_pos_embed
@HEADS.register_module()
class MultiScaleDecoder(BaseDecodeHead):

    def __init__(self, resize_factors = None, **kwargs):
        super().__init__(**kwargs)
        # assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(384)
        self.resize_factors = resize_factors

        in_out = [
            [768, 384],
            [384, 192],
            [192, 96],
        ]

        self.deconv_layers = nn.ModuleList([
                nn.ConvTranspose2d(in_out[i][0],in_out[i][1], kernel_size=3, stride=2)
                for i in range(3)
        ])

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        # print("inputs", [i.shape for i in inputs])

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            # print("resize_factors", self.resize_factors)


            upsampled_inputs = []

            for input in inputs:
                for deconv_layer in self.deconv_layers:
                    input = deconv_layer(input)
                upsampled_inputs.append(input)
            
            

            
            inputs = torch.cat(upsampled_inputs, dim=1)

        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    


@HEADS.register_module()
class TransformerDecoder(BaseDecodeHead):

    def __init__(self,img_size,embed_dim,decoder_embed_dim,patch_size,decoder_depth,classes,num_heads,drop,attn_drop,drop_path,multiout, resize_factors = None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) *  (img_size[1] // patch_size)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim,num_heads = num_heads,drop = drop, attn_drop = attn_drop, drop_path=drop_path, attn_class=MemEffAttention)
            for i in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * classes, bias=True) # decoder to patch
        self.classes = classes
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.initialize_weights()
        self.multiout = multiout

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
       
      

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._forward_decode(inputs)

        # print("feats", feats.shape)
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.classes))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.classes, h * p, h * p))
        return imgs

    def _forward_decode(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        # print("inputs", [i.shape for i in inputs])
        if self.multiout:
            new_inputs = []
            for (y_1, y_2) in inputs:
                new_input = torch.cat((y_1, y_2), dim=1)
                new_inputs.append(new_input)
                
            inputs = new_inputs
        
        inputs = torch.cat(inputs, dim=1)
        inputs = inputs.view(-1,self.embed_dim,self.num_patches)
        inputs = inputs.transpose(1, 2)
        batch_size = inputs.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token to match the batch size
        inputs = torch.cat([cls_tokens, inputs], dim=1)
        # print("inputs shape: ",inputs.shape)
        x = self.decoder_embed(inputs)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        x = self.unpatchify(x)

        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # print(output.shape)
        output = self.cls_seg(output)
        # print(output.shape)
        return output
    



@HEADS.register_module()
class HSIBNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None,multiout=False, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.multiout = multiout

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        
        if self.multiout:
            new_inputs = []
            for (y_1, y_2) in inputs:
                new_input = torch.cat((y_1, y_2), dim=1)
                new_inputs.append(new_input)
                
            inputs = new_inputs
        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]

                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
            # print(inputs.shape)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        # print("inputs", [i.shape for i in inputs])
        return inputs

    def forward(self, inputs):
        """Forward function."""
        print(inputs)
        output = self._forward_feature(inputs)
        # print("output", output.shape)
        output = self.cls_seg(output)
        # print("output after cls", output.shape)
        return output
