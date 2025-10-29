import warnings

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub

from models.common import Bottleneck, Concat, C3, SPPF, Conv
from models.yolo import Detect, Segment

from ... quant_wrappers import QuantStubWrapper, DeQuantStubWrapper, StubWrappable


class QuantizableConv(Conv, StubWrappable):
    def __init__(self, orig_module: Conv):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

    def insertQuantStub(self):
        self.conv = QuantStubWrapper(self.conv)

    def insertDeQuantStub(self):
        self.act = DeQuantStubWrapper(self.act)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, orig_module: Bottleneck):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        
        self.f_add = FloatFunctional()

    def forward(self, x):
        # OG: return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

        if self.add:
            return self.f_add.add(x, self.cv2(self.cv1(x)))
        else:
            return self.cv2(self.cv1(x))


class QuantizableConcat(Concat):
    def __init__(self, orig_module: Concat):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        
        self.q_cat = FloatFunctional()

    def forward(self, x):
        # OG: return torch.cat(x, self.d)

        return self.q_cat.cat(x, self.d)
    

class QuantizableC3(C3, StubWrappable):
    def __init__(self, orig_module: C3):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        
        self.q_cat = FloatFunctional()

    def forward(self, x):
        # OG: return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

        return self.cv3(
                self.q_cat.cat(
                    # (self.m(self.cv1(x)), self.cv2(x)), 1))
                    [self.m(self.cv1(x)), self.cv2(x)], 1))
    
    def insertQuantStub(self):
        self.cv1 = QuantStubWrapper(self.cv1)
        self.cv2 = QuantStubWrapper(self.cv2)

    def insertDeQuantStub(self):
        self.cv3 = DeQuantStubWrapper(self.cv3)


class QuantizableSPPF(SPPF, StubWrappable):
    def __init__(self, orig_module: SPPF):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        
        self.q_cat = FloatFunctional()

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y1 = self.m(x)
            Y2 = self.m(Y1)
            RESULT = self.cv2(
                        torch.cat(
                            (x, Y1, Y2, self.m(Y2)), 1))
        return RESULT
    
    def insertQuantStub(self):
        self.cv1 = QuantStubWrapper(self.cv1)

    def insertDeQuantStub(self):
        self.cv2 = DeQuantStubWrapper(self.cv2)


class QuantizableDetect(Detect, StubWrappable):
    def __init__(self, orig_module: Detect):
        self.__dict__ = orig_module.__dict__
        self.__dict__["type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # To hide typing warnings
        assert not orig_module.stride is None
        self.stride = orig_module.stride
        
        self.sigmoid = torch.nn.Sigmoid()

        # self.xy_mul_scalar = FloatFunctional()
        # self.xy_add = FloatFunctional()
        # self.xy_mul = FloatFunctional()
        # self.wh_mul = FloatFunctional()
        # self.wh_mul_scalar = FloatFunctional()
        # self.wh_mul2 = FloatFunctional()
        # self.q_cat = FloatFunctional()
        # self.stride_quant = QuantStub()
        # self.grid_quant = QuantStub()
        # self.anchor_grid_quant = QuantStub()
        
    def forward(self, x):   
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    if x[1].is_quantized:
                        print(f"[QuantizableDetect] Квантование для сегментации не реализовано!") # TODO use Logger

                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    x[i] = self.sigmoid(x[i])
                    xy, wh, conf = x[i].split((2, 2, self.nc + 1), 4)
                    # conf = torch.dequantize(conf)
            # Reshaping Output ----------------------------------------------------------------------------------------                          
                    ##### xy = (xy * 2 + self.grid[i]) * self.stride[i] #####
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    # xy = self.xy_mul_scalar.mul_scalar(xy, 2)                   
                    # xy = self.xy_add.add(xy, self.grid_quant(self.grid[i]))
                    # xy = self.xy_mul.mul(xy, self.stride_quant(self.stride[i]))                    
                    # xy = torch.dequantize(xy)
                    ##### wh = (wh * 2) ** 2 * self.anchor_grid[i] #####
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    # wh = self.wh_mul.mul(wh, wh)
                    # wh = self.wh_mul_scalar.mul_scalar(wh, 4)
                    # wh = self.wh_mul2.mul(wh, self.anchor_grid_quant(self.anchor_grid[i]))
                    # wh = torch.dequantize(wh)

                    # y = self.q_cat.cat((xy, wh, conf), 4)
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
            #----------------------------------------------------------------------------------------------------------
        if self.training:
            return x 
        elif self.export:
            return (torch.cat(z, 1),)
        else:
            return (torch.cat(z, 1), x)

    def insertQuantStub(self):
        for i in range(self.nl):
            self.m[i] = QuantStubWrapper(self.m[i])
    
    def insertDeQuantStub(self):
        for i in range(self.nl):
            self.m[i] = DeQuantStubWrapper(self.m[i])


_adaptable_types = (Bottleneck, Concat, C3, SPPF, Conv, Detect, nn.SiLU)
_activation_types = (nn.SiLU)
_mapping = {
    # -- Modules --------------------------
    Conv: QuantizableConv,
    Bottleneck: QuantizableBottleneck,
    Detect: QuantizableDetect,
    Concat: QuantizableConcat,
    C3: QuantizableC3,
    SPPF: QuantizableSPPF,
    # -- Activations ----------------------
    nn.SiLU: nn.Hardswish,
}


# TODO compare with https://github.com/pytorch/pytorch/blob/v2.8.0/torch/ao/quantization/quantize.py#L739
def swap_with_quantizables(module: nn.Module, mapping: dict = _mapping, qconfig=None, full_name="", verbose=False):
    for n, m in module.named_children():
        _full_name = f"{full_name}.{n}"
        
        # if type(m) in _adaptable_types:
        if type(m) in mapping.keys():
            if isinstance(m, _activation_types):
                qm = mapping[type(m)](inplace = m.inplace)
            else:
                qm = mapping[type(m)](m)

            if not qconfig is None:
                qm.qconfig = qconfig

            setattr(module, n, qm)

            if verbose:
                print(f"[swap_with_quantizables] {f'{_full_name:.30s}': <30}:  {type(m).__name__: ^20}  -->  {type(qm).__name__: ^20}")
        
        if len(list(m.children())) > 0:
            swap_with_quantizables(m, mapping, qconfig, _full_name, verbose)
