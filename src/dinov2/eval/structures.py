"""
------------------------------------------------------------------------------
Code copied over from Detectron2 for Mask2Former processing pipeline
------------------------------------------------------------------------------
"""

import torch
from typing import Any, List, Optional, Tuple, Union,Dict
import numpy as np
from torch import device
from torch import nn
from torchvision.ops import roi_align
import itertools
import warnings




class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor



class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.bool)
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.bool, device=torch.device("cpu"))
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasks":
        return BitMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].unsqueeze(0))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks(m)

    @torch.jit.unused
    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    # @staticmethod
    # def from_polygon_masks(
    #     polygon_masks: Union["PolygonMasks", List[List[np.ndarray]]], height: int, width: int
    # ) -> "BitMasks":
    #     """
    #     Args:
    #         polygon_masks (list[list[ndarray]] or PolygonMasks)
    #         height, width (int)
    #     """
    #     if isinstance(polygon_masks, PolygonMasks):
    #         polygon_masks = polygon_masks.polygons
    #     masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
    #     if len(masks):
    #         return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))
    #     else:
    #         return BitMasks(torch.empty(0, height, width, dtype=torch.bool))

    # @staticmethod
    # def from_roi_masks(roi_masks: "ROIMasks", height: int, width: int) -> "BitMasks":
    #     """
    #     Args:
    #         roi_masks:
    #         height, width (int):
    #     """
    #     return roi_masks.to_bitmasks(height, width)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        It has less reconstruction error compared to rasterization with polygons.
        However we observe no difference in accuracy,
        but BitMasks requires more memory to store all the masks.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )
        output = output >= 0.5
        return output

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        """
        Concatenates a list of BitMasks into a single BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])

        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks
    



class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        from torchvision import __version__

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized:
            input = input.dequantize()
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__