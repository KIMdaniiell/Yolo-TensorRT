import enum
import time
import contextlib

import torch
from typing import Callable, Any, Dict, Sequence, Optional

def has_qconfig(
        module: torch.nn.Module, 
        full_name="model", 
        recursive=True, 
        reverse=False,
        callbacks: Sequence[Callable[[torch.nn.Module], None]] = [],
        verbose=False):

    result = False

    if (hasattr(module, "qconfig") and module.qconfig is not None) ^ (reverse):
        if verbose:
            print(f"[hasqconfig]  {f'{full_name:.30s}': <30} ({type(module).__name__})")
        for callback in callbacks:
            callback(module)
        result = True

    if recursive and (callbacks or verbose or not result):
        for n, m in module.named_children():
            _full_name = f"{full_name}.{n}"      
            result = has_qconfig(m, _full_name, recursive, reverse, callbacks, verbose) or result

    return result


def propagate_qconfig(module: torch.nn.Module, qconfig):
    module.qconfig = qconfig
    for child in module.children():
        propagate_qconfig(child, qconfig)


class MeasurementUnits(enum.Enum):
    def __call__(self, d: int | float) -> float:
        return d/self.value
    
    @classmethod
    def auto_select_unit(cls, value: int|float):
        value = abs(value)
        
        final_unit = next(cls.__iter__())
        for unit in cls:
            if final_unit is None:
                final_unit = unit
                continue

            if unit.value / 2 > value: 
                return final_unit
            final_unit = unit
        return final_unit
    
    @classmethod
    def verbose(cls, value: int | float, unit: Optional['MeasurementUnits'] = None) -> str:
        if unit is None:
            unit = cls.auto_select_unit(value)
        if unit is None:
            return f"{value}"    
        return f"{unit(value): .3f} {unit.name}"
    


class TimeUnits(MeasurementUnits):
    MCS = 1
    MS = 1_000 * MCS
    S = 1_000 * MS
    M = 60 * S

class MemoryUnits(MeasurementUnits):
    # BIT = 1
    # BYTE = 8 * BIT
    BYTE = 1
    B = BYTE
    KB = 1000 * B
    MB = 1000 * KB
    GB = 1000 * MB


class Timer(contextlib.ContextDecorator):
    def __init__(
            self, 
            message: Optional[str] = None):
        if not (message is None):
            self.message = message

    def __enter__(self):
        self.start = time.time_ns()
        return self

    def __exit__(self, type, value, traceback):
        t = (time.time_ns() - self.start) / 1000 # to mcs

        if hasattr(self, "message"):
            print(f"{self.message} [{TimeUnits.S(t): .3f} s]")


class Dictionaries:

    @staticmethod
    def _filter_keys(
        d: dict, 
        predicat: Callable[[Any, Any], bool]
        ) -> Dict[Any, Any]:

        return {k: v for k, v in d.items() if predicat(k, v)}
    

    # In-place version of `_filter_by_key(...)`
    @staticmethod
    def _filter_keys_(
        d: dict, 
        predicat: Callable[[Any, Any], bool]
        ) -> Dict[Any, Any]:

        for k, v in d.items():
            if not predicat(k, v): 
                del d[k]
        return d
    

    @staticmethod
    def filter_keys(
        dictionary: dict, 
        predicat: Callable[[Any, Any], bool] = lambda key, _: not (key is None),
        in_place: bool = False
        ) -> Dict[Any, Any]:

        if dictionary is None: return None

        if predicat is None: return dictionary

        if in_place:
            return Dictionaries._filter_keys_(dictionary, predicat)
        else:
            return Dictionaries._filter_keys(dictionary, predicat)


    @staticmethod
    def select_keys(
        dictionary: dict,
        include_keys: Optional[Sequence[Any]] = None,
        exclude_keys: Optional[Sequence[Any]] = None,
        in_place: bool = False
    ) -> Dict[Any, Any]:
        
        assert not (include_keys is None and exclude_keys is None), """
        Both `include_keys` and `exclude_keys` can't be None.
        Either one must be defined."""
        
        include_predicat: Callable[[Any, Any], bool] = lambda key, _: key in include_keys
        exclude_predicat: Callable[[Any, Any], bool] = lambda key, _: not (key in exclude_keys)
        
        if include_keys is None:
            predicat = exclude_predicat
        elif exclude_keys is None: 
            predicat = include_predicat
        else:
            include_keys = [ik for ik in include_keys if not(ik in exclude_keys)]
            predicat = include_predicat

        return Dictionaries.filter_keys(dictionary, predicat, in_place)
