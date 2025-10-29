from __future__ import annotations
import abc
import os
import json
from typing import Sequence, List, Dict, Optional, Any

from tabulate import tabulate
from torch.profiler._memory_profiler import Category, _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from .general import Dictionaries, MemoryUnits, TimeUnits


class Report(abc.ABC):
    headers: List[str]
    reports: List[Report] = []

    @abc.abstractmethod
    def __init__(
                self, 
                data: Sequence, 
                title: Optional[str] = None
            ) -> None: 
        pass

    @abc.abstractmethod
    def to_list(
                self, 
                include_title: bool = True
            ) -> List: 
        pass

    def set(self, name: str, value: Any):
        self.__setattr__(name, value)
    
    def record(self):
        self.__class__.reports.append(self)
        return self
    
    @classmethod
    def to_table(
                cls,
                report_data: Optional[Report | Sequence[Report]] = None, 
                **kwargs
            ) -> str:
        if report_data is None:
            report_data = cls.reports
        elif isinstance(report_data, Report):
            report_data = [report_data]
                
        tabular_data: List[List] = [r.to_list() for r in report_data]
            
        return tabulate(
            tabular_data=tabular_data,
            headers=cls.headers,
            **kwargs)


class Yolov5ValidationReport(Report):
    headers = ["Stage" ,"P", "R", "mAP50", "mAP50-95", "Pre-process per image", "Inference per image", "NMS per image", "Max Memory", "Sum Memory"]
    memory_unit = MemoryUnits.MB
    time_unit = TimeUnits.MS
    
    def __init__(   
                self, 
                results: tuple = ((0.,0.,0.,0.), (), (0.,0.,0.)), 
                title: str = "", 
            ) -> None:
        self.title = title
            
        perfomance = results[0]
        self.precision = perfomance[0]
        self.recall    = perfomance[1]
        self.map50     = perfomance[2]
        self.map50_95  = perfomance[3]

        timings = results[2] # by default val.py returns timings in ms
        # ...*1000 - to convert time data to TimeUnits.MCS 
        self.preprocess_t = timings[0] * 1000
        self.inference_t  = timings[1] * 1000
        self.nms_t        = timings[2] * 1000

    def to_list(self, include_title: bool = True) -> List:
        result = [self.precision, self.recall, self.map50, self.map50_95,
                  TimeUnits.verbose(self.preprocess_t, self.time_unit), 
                  TimeUnits.verbose(self.inference_t, self.time_unit), 
                  TimeUnits.verbose(self.nms_t, self.time_unit)]
        if include_title:
            result.insert(0, self.title)
        if hasattr(self, "max_memory"):
            result.append(MemoryUnits.verbose(self.__getattribute__("max_memory"), self.memory_unit))
        if hasattr(self, "sum_memory"):
            result.append(MemoryUnits.verbose(self.__getattribute__("sum_memory"), self.memory_unit))
        return result


class ProfilingData:
    FILE_NAME: str = "profiling_results.json"

    def __init__(self, path: str, ticks_lim: Optional[int] = None,):
        ProfilingData._check_profiling_results(path)
        self.path = path

        timestamps, memory = ProfilingData._load_profiling_results(
            os.path.join(path, ProfilingData.FILE_NAME))
        
        # Trimming data arrays length
        if ticks_lim is None:
            ticks_lim = len(timestamps)
        timestamps = timestamps[:ticks_lim]
        memory = memory[:ticks_lim]

        timestamps = [t - timestamps[0] for t in timestamps]
        self.timestamps:List[int] = timestamps
        self.memory_by_category:Dict[Category, List[int]] = self._get_memory_by_category(memory)
        self.max_memory:int = self._get_max_memory()
        self.sum_memory:int = self._get_sum_memory()
    
    def _get_memory_by_category(
            self, 
            memory: List[List]
            ) -> Dict[Category, List[int]]:
        memory_by_category = {c: [] for c, _ in _CATEGORY_TO_INDEX.items()}
        for m in memory:
            for category, i in _CATEGORY_TO_INDEX.items():
                memory_by_category[category].append(m[i+1])
        return memory_by_category
    
    def _get_max_memory(self) -> int:
        max_memory = 0
        for m in zip(*(memory for memory in self.memory_by_category.values())):
            max_memory = max(max_memory, sum(m))
        return max_memory

    def _get_sum_memory(self) -> int:
        sum_memory = 0
        for m in zip(*(memory for memory in self.memory_by_category.values())):
            sum_memory += sum(m)
        return sum_memory

    def filter_categories(
            self,
            include_categories: Optional[Sequence[Category]] = None,
            exclude_categories: Optional[Sequence[Category]] = None
            ) -> None:
        if not (include_categories is None and exclude_categories is None):
            self.memory_by_category = Dictionaries.select_keys(self.memory_by_category, 
                                                    include_categories, 
                                                    exclude_categories)
            self.max_memory = self._get_max_memory()

    @staticmethod
    def _check_profiling_results(path:str):
        assert not (path is None), "Path to profiling results is not defined"
        assert os.path.exists(path) and os.path.isdir(path), "Profiling results dir is not defined"

        profiling_results_path = os.path.join(path, ProfilingData.FILE_NAME)
        assert os.path.exists(profiling_results_path), f"Profiling results dir does not contain {ProfilingData.FILE_NAME}"

    @staticmethod
    def _load_profiling_results(path:str):
        with open(path) as f:
            (timestamps, memory) = json.load(f)
        return (timestamps, memory)


# Reference - https://pytorch-hub-preview.netlify.app/blog/understanding-gpu-memory-1/

def _plot_profiling_results_join(
        ax: Axes,

        timestamps:List[int], 
        memory_by_category:Dict[Category, List[int]], 

        time_unit:TimeUnits = TimeUnits.MS,
        memory_unit:MemoryUnits = MemoryUnits.GB
        ) -> int:
    
    sum_memory = [0 for _ in timestamps]

    for category, memory in memory_by_category.items():

        # Заливка
        ax.fill_between(
            x = timestamps, 
            y1 = sum_memory, 
            y2 = [_c + _m for (_c,_m) in zip(sum_memory, memory)],
            where = [m != 0 for m in memory],
            
            color=_CATEGORY_TO_COLORS[category],
            alpha=0.5,
            label="Unknown" if category is None else category.name)
        # Линия
        # ax.plot(
        #     timestamps, 
        #     sum_memory, 
        #     color=_CATEGORY_TO_COLORS[category])

        for j, m in enumerate(memory):
            sum_memory[j] += m

    # Подписи оси абсцисс (Ox)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(time_unit(x)):_d}"))  # 100_000
    ax.set_xlabel(f"Time ({time_unit.name.lower()})")
    # Подписи оси ординат (Oy)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{memory_unit(y)}"))
    ax.set_ylabel(f"Memory ({memory_unit.name})")

    ax.set_title(label = f"Max memory: {memory_unit(max(sum_memory)):.2f} {memory_unit.name}",
                 fontdict = {'fontsize': 10})
    ax.legend()
    ax.grid(True)
    return max(sum_memory)

def _plot_profiling_results_separatly(
        axes: List[Axes],

        timestamps:List[int], 
        memory_by_category:Dict[Category, List[int]],
        max_memory: Optional[int] = None,

        time_unit:TimeUnits = TimeUnits.MS,
        memory_unit:Optional[MemoryUnits] = None,
        ) -> None:
    
    for ax, (category, memory) in zip(axes, memory_by_category.items()):
    
        # Заливка
        ax.fill_between(
            x = timestamps,
            y1 = memory,
            color=_CATEGORY_TO_COLORS[category],
            alpha=0.5,
            label="Unknown" if category is None else category.name)
        # Линия
        # ax.plot(
        #     timestamps,
        #     memory, 
        #     color=_CATEGORY_TO_COLORS[category])
        
        # Подписи оси абсцисс (Ox)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{int(time_unit(x)):_d}"))
        ax.set_xlabel(f"Time ({time_unit.name.lower()})")
        # Подписи оси ординат (Oy)
        if not (max_memory is None):
            ax.set_ylim(top=max_memory)
        def create_fmtr(v, mu):
            if mu is None:
                memory_unit = MemoryUnits.auto_select_unit(v)
            fmtr = FuncFormatter(lambda y, _: f"{memory_unit(y):.2f}")
            return (fmtr, memory_unit)
        fmtr, mu = create_fmtr(max((abs(v) for v in ax.get_ylim())), memory_unit)
        ax.yaxis.set_major_formatter(fmtr)
        ax.set_ylabel(f"Memory ({mu.name})")

        ax.set_title(label = f"Max memory: {mu(max(memory)):.2f} {mu.name}",
                    fontdict = {'color': _CATEGORY_TO_COLORS[category], 'fontsize': 10})
        ax.legend()
        ax.grid(True)

def plot_profiling_results(
        profiling_data: ProfilingData,

        include_categories: Optional[Sequence[Category]] = None,
        exclude_categories: Optional[Sequence[Category]] = None,

        time_lim: Optional[int] = None,
        save_dir: Optional[str] = None,
        show: bool = True,
        time_unit:TimeUnits = TimeUnits.MS,
        memory_unit:MemoryUnits = MemoryUnits.GB,
        ) -> None:
    
    # Trimming data arrays length
    if time_lim is None:
        time_lim = len(profiling_data.timestamps)
    timestamps = profiling_data.timestamps[:time_lim]
    memory_by_category = {
        category: memory[:time_lim] 
        for category, memory in profiling_data.memory_by_category.items()}
    
    # Filtering categoties
    if not (include_categories is None and exclude_categories is None):
        memory_by_category = Dictionaries.select_keys(memory_by_category, 
                                                    include_categories, 
                                                    exclude_categories)
    
    fig = plt.figure(figsize=(6.4*3, 1.6*8))

    # Plotting the results combined into a single graph
    ax = plt.subplot2grid((len(_CATEGORY_TO_INDEX), 3), (0,0), rowspan=4)
    max_memory = _plot_profiling_results_join(
        ax,
        timestamps,
        memory_by_category,
        time_unit,
        memory_unit)
    
    # Plotting the results into separate graphs
    axes = [plt.subplot2grid(
        shape = (len(_CATEGORY_TO_INDEX), 3), 
        loc = (i%4, 1 + i//4)) for i in range(len(_CATEGORY_TO_INDEX))]
    _plot_profiling_results_separatly(
        axes,
        timestamps,
        memory_by_category,
        # max_memory,
        time_unit=time_unit)

    fig.suptitle(f"Max memory: {memory_unit(max_memory):.2f} {memory_unit.name}", fontsize = 20)

    plt.subplots_adjust(top=0.97, bottom=0.06, hspace=1)
    plt.tight_layout()
    
    if not (save_dir is None):
        plt.savefig(os.path.join(save_dir, "memory_profiling.png"))

    if show:
        plt.show()
