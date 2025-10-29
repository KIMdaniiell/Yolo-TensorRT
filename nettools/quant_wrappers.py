import torch
from torch.ao.quantization import QuantStub, DeQuantStub
import abc


class QuantStubWrapper(torch.nn.Module):
    quant: QuantStub
    module: torch.nn.Module
    
    def __init__(self, module: torch.nn.Module):
        assert not isinstance(module, QuantStubWrapper), "Module already wrapped with QuantStubWrapper"

        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("quant", QuantStub(qconfig))
        self.add_module("module", module)
        self.train(module.training)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.quant(X)
        return self.module(X)


class DeQuantStubWrapper(torch.nn.Module):
    dequant: DeQuantStub
    module: torch.nn.Module

    def __init__(self, module: torch.nn.Module):
        assert not isinstance(module, DeQuantStubWrapper), "Module already wrapped with DeQuantStubWrapper"

        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("module", module)
        self.add_module("dequant", DeQuantStub(qconfig))
        self.train(module.training)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.module(X)
        return self.dequant(X)


class StubWrappable(abc.ABC):

    @abc.abstractmethod
    def insertQuantStub(self): pass

    @abc.abstractmethod
    def insertDeQuantStub(self): pass
