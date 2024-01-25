import torch
from torch import nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

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
        print("x", x.shape)
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
        print("inputs", [i.shape for i in inputs])

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
            print("resize_factors", self.resize_factors)


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