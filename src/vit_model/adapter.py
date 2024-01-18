import torch
from torch import nn

class InputAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,n_stepdowns=3, bias=False):
        super().__init__()
        step_size = (in_channels - out_channels)//n_stepdowns

        out_levels = torch.arange(out_channels,in_channels,step_size,dtype=torch.int)
        out_levels = torch.flip(out_levels,dims=(0,))[1:]

        self.conv_layers = nn.ModuleList()
        for i in range(n_stepdowns):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_levels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
            in_channels = out_levels[i]
        print(self.conv_layers)

    def forward(self,x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class OutputAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,n_stepups=3, bias=False):
        super().__init__()
        step_size = (out_channels - in_channels)//n_stepups

        out_levels = torch.arange(in_channels,out_channels,step_size,dtype=torch.int)
        out_levels = out_levels[1:]
        out_levels[-1] = out_channels

        self.conv_layers = nn.ModuleList()
        for i in range(n_stepups):
            self.conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_levels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
            in_channels = out_levels[i]
        print(self.conv_layers)

    def forward(self,x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ModelAdapter(nn.Module):
    def __init__(self, model, input_adapter, output_adapter):
        super().__init__()
        self.model = model
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    def forward(self,x):
        x = self.input_adapter(x)
        x,mask = self.model(x)
        x = self.output_adapter(x)
        return x,mask

