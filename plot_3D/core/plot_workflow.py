import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from plot_3D.core.plot_3D_params_space_plt import *  # 假设这些模块存在
from plot_3D.advance_plot_styles.polar_plot import plot_polar_line
from plot_3D.core.utils import *  # load_lumerical_jsondata 等
from plot_3D.advance_plot_styles.scatter_plot import plot_scatter_advanced


@dataclass
class PlotConfig:
    """配置类：统一参数管理"""
    figsize: tuple = (3, 4)
    fs: int = 16  # 字体大小
    save_dir: str = './rsl'
    show: bool = True
    plot_params: Dict[str, Any] = None  # e.g., {'cmap': 'magma', 'add_colorbar': True}
    annotations: Dict[str, Any] = None  # e.g., {'xlabel': r'f (c/P)', 'ylabel': 'P'}
    dpi: int = 300
    tick_direction: str = 'out'  # 刻度线方向


class BasePlotter(ABC):
    """
    基类：提取所有脚本共性（加载→准备→计算数据→绘图→注解→保存）
    关键：prepare_data抽象，用户手动重写；compute_xxx返回数据，便于main后处理
    使用：main中 load → prepare → compute_xxx（返回数据） → 后处理 → plot_xxx（绘图） → add_annotations → save
    支持重叠：re_initialized只重置data，不碰fig/ax
    """

    def __init__(self, config: Optional[Union[PlotConfig, Dict]] = None, data_path: Optional[str] = None):
        self.config = PlotConfig(**config) if isinstance(config, dict) else config or PlotConfig()
        self.data_path = data_path
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.raw_dataset: Any = None
        self.x_vals: Optional[np.ndarray] = None
        self.y_vals: Optional[np.ndarray] = None
        self.subs: Optional[List[np.ndarray]] = None
        plt.rcParams.update({'font.size': config.fs})
        plt.rcParams['xtick.direction'] = config.tick_direction  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = config.tick_direction  # 将y轴的刻度方向设置向内

    def re_initialized(self, config: Optional[Union[PlotConfig, Dict]] = None, data_path: Optional[str] = None) -> None:
        """优化：只重置config/data相关，不重置fig/ax，支持重叠绘图"""
        self.config = PlotConfig(**config) if isinstance(config, dict) else config or self.config
        self.data_path = data_path or self.data_path
        self.raw_dataset = None
        self.x_vals = None
        self.y_vals = None
        self.subs = None
        print("Re-initialized data/config，fig/ax保留以支持重叠绘图 🔄")
        return self

    def load_data(self) -> None:
        """共性：加载，支持JSON/Pickle（用户可重写自定义加载）"""
        if not self.data_path:
            raise ValueError("data_path 未提供！")
        if self.data_path.endswith('.json'):
            self._load_json()
        elif self.data_path.endswith('.pkl'):
            self._load_pickle()
        else:
            raise ValueError(f"不支持的文件类型: {self.data_path}")
        print(f"数据加载成功 📂")

    def _load_json(self) -> None:
        """JSON加载骨架（脚本1专用，用户重写扩展多文件）"""
        self.raw_dataset = load_lumerical_jsondata(self.data_path)

    def _load_pickle(self) -> None:
        """Pickle加载骨架（脚本2/3/4，用户重写后处理）"""
        with open(self.data_path, 'rb') as f:
            self.raw_dataset = pickle.load(f)
        self.x_vals = self.raw_dataset.get('x_vals', np.array([]))
        self.y_vals = self.raw_dataset.get('y_vals', np.array([]))
        self.subs = self.raw_dataset.get('subs', [])
        print(f"Pickle基础提取: x_shape={self.x_vals.shape}, subs_len={len(self.subs)} 🔍")

    @abstractmethod
    def prepare_data(self) -> None:
        """抽象：最灵活部分！用户手动重写：提取键、过滤NaN、计算衍生"""
        pass

    @abstractmethod
    def plot(self) -> None:
        """抽象：留空，用户在main手动调用绘图方法（如plot_line）"""
        pass

    def twin_plot_ax(self, twinx: bool = False, twiny: bool = False) -> plt.Axes:
        if twiny:
            if not hasattr(self, 'twiny_ax'):
                self.twiny_ax = self.ax.twiny()
            return self.twiny_ax
        if twinx:
            if not hasattr(self, 'twinx_ax'):
                self.twinx_ax = self.ax.twinx()
            return self.twinx_ax
        else:
            return self.ax

    def new_fig(self, projection: str = 'rectilinear') -> None:
        """共性：创建新fig/ax，支持polar。手动调用以控制新图"""
        kwargs = {'figsize': self.config.figsize}
        if projection == 'polar':
            kwargs['subplot_kw'] = {'projection': 'polar'}
        self.fig, self.ax = plt.subplots(**kwargs)

    def add_annotations(self) -> None:
        """共性：添加标签/限（用户可重写加自定义scale）"""
        if self.config.annotations is None:
            print("Warning: 未设置annotations ⚠️")
        self.fig, self.ax = add_annotations(self.ax, self.config.annotations)

    def add_twinx_annotations(self) -> None:
        """共性：添加双轴标签"""
        if self.config.annotations is None:
            print("Warning: 未设置annotations ⚠️")
        self.fig, self.twinx_ax = add_annotations(self.twinx_ax, self.config.annotations)

    def add_twiny_annotations(self) -> None:
        """共性：添加双轴标签"""
        if self.config.annotations is None:
            print("Warning: 未设置annotations ⚠️")
        self.fig, self.twiny_ax = add_annotations(self.twiny_ax, self.config.annotations)

    def save_and_show(self, save=True, save_type='svg', custom_name: Optional[str] = None,
                      custom_abs_path: Optional[str] = None) -> None:
        """共性：保存/show（支持自定义名）"""
        if save:
            full_params = self.config.plot_params or {}
            if custom_abs_path:
                image_path = custom_abs_path
            elif custom_name:
                image_path = os.path.join(self.config.save_dir, custom_name)
            else:
                image_path = generate_save_name(self.config.save_dir, full_params)
            plt.savefig(image_path + f'.{save_type}', dpi=self.config.dpi, bbox_inches="tight", transparent=True)
            print(f"图像已保存为：{image_path} 🎨")
            plt.savefig('temp_output.svg', dpi=self.config.dpi, bbox_inches="tight", transparent=True)
            print("Temp figure saved as 'temp_output.svg'.")
        if self.config.show:
            plt.show()

    def run_full(self) -> None:
        """可选完整链：但不推荐，用手动链代替"""
        self.load_data()
        self.prepare_data()
        self.new_fig()
        self.plot()
        self.add_annotations()
        self.save_and_show()
        print("全流程完成！🚀")


# 分类子类0: ScatterPlotter（0D点类）
class ScatterPlotter(BasePlotter):
    """0D点类骨架：提供plot_scatter绘图"""

    def plot_scatter(self, x: np.ndarray, z1: np.ndarray, **kwargs) -> None:
        """辅助：通用plot_scatter_advanced"""
        params = {**self.config.plot_params, **kwargs}
        ax = self.twin_plot_ax(kwargs.get('twinx', False), kwargs.get('twiny', False))
        self.ax = plot_scatter_advanced(ax, x, z1=z1, z3=z1, **params)


# 分类子类1: LinePlotter（1D线类）
class LinePlotter(BasePlotter):
    """1D线类骨架：提供plot_line绘图，用户在main调用"""

    def plot_line(self, x: np.ndarray, z1: np.ndarray, **kwargs) -> None:
        """辅助：通用plot_line_advanced，支持twin轴"""
        params = {**self.config.plot_params, **kwargs}
        ax = self.twin_plot_ax(kwargs.get('twinx', False), kwargs.get('twiny', False))
        self.ax = plot_line_advanced(ax, x, z1=z1, **params)


# 分类子类2: PolarPlotter（极坐标）
class PolarPlotter(BasePlotter):
    """极坐标骨架：提供plot_polar绘图"""

    def plot_polar(self, theta: np.ndarray, radial: np.ndarray, **kwargs) -> None:
        """辅助：通用plot_polar_line"""
        params = {**self.config.plot_params, **kwargs}
        self.ax = plot_polar_line(self.ax, theta, radial, **params)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_thetalim(np.deg2rad(-60), np.deg2rad(60))


# 分类子类3: HeatmapPlotter（2D类）
class HeatmapPlotter(BasePlotter):
    """2D类骨架：提供plot_heatmap/multiline绘图"""

    def plot_heatmap(self, Z: np.ndarray, x_vals=None, y_vals=None, **kwargs) -> None:
        """辅助：plot_2d_heatmap"""
        params = {**self.config.plot_params, **kwargs}
        if x_vals is None:
            x_vals = np.arange(Z.shape[0])
        if y_vals is None:
            y_vals = np.arange(Z.shape[1])
        self.fig, self.ax = plot_2d_heatmap(self.ax, x_vals, y_vals, Z, params)

    def plot_multiline_2d(self, Z: np.ndarray, x_vals=None, y_vals=None, **kwargs) -> None:
        """辅助：plot_2d_multiline"""
        params = {**self.config.plot_params, **kwargs}
        if x_vals is None:
            x_vals = np.arange(Z.shape[0])
        if y_vals is None:
            y_vals = np.arange(Z.shape[1])
        self.fig, self.ax = plot_2d_multiline(self.ax, x_vals, y_vals, Z, params)
