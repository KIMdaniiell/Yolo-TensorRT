import torch.nn as nn

from models.common import Conv
from models.yolo import BaseModel
from utils.torch_utils import fuse_conv_and_bn


class ConvFused(Conv):
    def __init__(self, orig_module: Conv):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        fused = not hasattr(orig_module, "bn")
        if not fused:
            self.conv = fuse_conv_and_bn(orig_module.conv, orig_module.bn)
            delattr(self, "bn")

    def forward(self, x):
        return self.act(self.conv(x))


_mapping = {Conv: ConvFused}

def swap_modules(module: nn.Module, mapping: dict = _mapping, full_name="", verbose=False):
    for n, m in module.named_children():
        _full_name = f"{full_name}.{n}"
        
        if type(m) in mapping.keys():
            new_m = mapping[type(m)](m)
            setattr(module, n, new_m)

            if verbose:
                print(f"[swap_modules] {f'{_full_name:.30s}': <30}:  {type(m).__name__: ^20}  -->  {type(new_m).__name__: ^20}")

        if len(list(m.children())) > 0:
            swap_modules(m, mapping, _full_name, verbose)


def fuse(self: BaseModel):
    print("Fusing layers... ")

    swap_modules(self)

    self.info()
    return self