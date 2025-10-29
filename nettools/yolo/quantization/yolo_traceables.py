import torch

from typing import List

from models.common import Concat
from models.yolo import Detect
from utils.general import check_version


class TraceableConcat(Concat):
    def __init__(self, orig_module: Concat):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # see https://github.com/pytorch/pytorch/issues/79715
        froms: List[int] = getattr(orig_module, "f")
        assert isinstance(froms, List), f"[TraceableConcat] OrigModule.f expected to be List of ints, but got {type(froms)}"
        self.NUM_OF_SECTIONS = len(froms) 

    def forward(self, x):
        lst = []
        for i in range(self.NUM_OF_SECTIONS):
            lst.append(x[i])
        return torch.cat(lst, self.d)


def _make_grid(anchors, stride, na, nx, ny, nl, torch_1_10=check_version(torch.__version__, "1.10.0")):
    """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""

    grid, anchor_grid = [], []
    for i in range(nl):
        d = anchors[i].device
        t = anchors[i].dtype
        shape = 1, na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid.append(torch.stack((xv, yv), 2).expand(shape) - 0.5)  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid.append((anchors[i] * stride[i]).view((1, na, 1, 1, 2)).expand(shape))
    return grid, anchor_grid

torch.fx.wrap("_make_grid")

def _forward(x, nl, na, no, nc, anchors, stride):
    z = []  # inference output

    for i in range(nl):
        bs, _, ny, nx, _ = x[i].shape
        # x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        d = anchors[i].device
        t = anchors[i].dtype
        shape = 1, na, ny, nx, 2  # grid shape

        _y, _x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, "1.10.0"):
            yv, xv = torch.meshgrid(_y, _x, indexing="ij")
        else:
            yv, xv = torch.meshgrid(_y, _x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5 
        anchor_grid = (anchors[i] * stride[i]).view((1, na, 1, 1, 2)).expand(shape)

        xy, wh, conf = x[i].sigmoid().split((2, 2, nc + 1), 4)
        xy = (xy * 2 + grid) * stride[i]  # xy
        wh = (wh * 2) ** 2 * anchor_grid  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, na * nx * ny, no))
    
    return (torch.cat(z, 1), x)

torch.fx.wrap("_forward")

# def _arange(end, device, dtype):
#     return torch.arange(end, device=device ,dtype=dtype)
# torch.fx.wrap("_arange")

# def _expand(shape, tensor):
#     return tensor.expand(shape)
# torch.fx.wrap("_expand")

# def _expand_anchor_grid(shape, anchors, stride, na):
#     # (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2))
#     return (anchors * stride).view((1, na, 1, 1, 2)).expand(shape)
# torch.fx.wrap("_expand_anchor_grid")

class TraceableDetect(Detect):
    def __init__(self, orig_module: Detect):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # To hide typing warnings
        # assert not orig_module.stride is None
        # self.stride = orig_module.stride
        
        # Making grid every time, because can't analyze the input shape.
        # See https://docs.pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace
        # self.dynamic = True

    def forward(self, x):
        x = [self.m[i](x[i]) for i in range(self.nl)]

        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        return _forward(x, self.nl, self.na, self.no, self.nc, self.anchors, self.stride)

    # def forward(self, x):
    #     x = [self.m[i](x[i]) for i in range(self.nl)]

    #     z = []  # inference output
    #     for i in range(self.nl):
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    #         # self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #         d = self.anchors[i].device
    #         t = self.anchors[i].dtype
    #         shape = 1, self.na, ny, nx, 2  # grid shape
    #         _y, _x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    #         if check_version(torch.__version__, "1.10.0"):
    #             yv, xv = torch.meshgrid(_y, _x, indexing="ij")
    #         else:
    #             yv, xv = torch.meshgrid(_y, _x)
    #         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    #         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)

    #         # Detect (boxes only)
    #         xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
    #         xy = (xy * 2 + grid) * self.stride[i]  # xy
    #         wh = (wh * 2) ** 2 * anchor_grid  # wh
    #         y = torch.cat((xy, wh, conf), 4)
    #         z.append(y.view(bs, self.na * nx * ny, self.no))

    #     return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    # def forward(self, x):
    #     z = []  # inference output
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         # bs, _, ny, nx = _shape(x[i])
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    #         if not self.training:  # inference
    #             if self.dynamic:
    #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #                 # self.grid[i], self.anchor_grid[i] = _make_grid(self.anchors, self.stride, self.na, nx, ny, i)
            
    #             xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
    #             xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
    #             wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
    #             y = torch.cat((xy, wh, conf), 4)
    #             z.append(y.view(bs, self.na * nx * ny, self.no))

    #     if self.training:
    #         return x 
    #     elif self.export:
    #         return (torch.cat(z, 1),)
    #     else:
    #         return (torch.cat(z, 1), x)
    
    # def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
    #     """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
    #     d = self.anchors[i].device
    #     t = self.anchors[i].dtype
    #     shape = 1, self.na, ny, nx, 2  # grid shape
    #     # y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    #     y, x = _arange(ny, device=d, dtype=t), _arange(nx, device=d, dtype=t)
    #     yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
    #     grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    #     anchor_grid = _expand_anchor_grid(shape, self.anchors[i], self.stride[i], self.na)
    #     return grid, anchor_grid
