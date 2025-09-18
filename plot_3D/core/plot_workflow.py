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

@dataclass
class PlotConfig:
    """配置类：统一参数管理"""
    figsize: tuple = (3, 4)
    save_dir: str = './rsl'
    show: bool = True
    plot_params: Dict[str, Any] = None  # e.g., {'cmap': 'magma', 'add_colorbar': True}
    annotations: Dict[str, Any] = None  # e.g., {'xlabel': r'f (c/P)', 'ylabel': 'P'}
    dpi: int = 300

class BasePlotter(ABC):
    """
    基类：提取所有脚本共性（加载→准备→绘图→注解→保存）
    关键：prepare_data() 和 plot() 抽象，用户在脚本类中手动重写
    使用：脚本中继承，链式调用 prepare_data() → plot_specific() → add_annotations() → save_and_show()
    """
    def __init__(self, config: Optional[Union[PlotConfig, Dict]] = None, data_path: Optional[str] = None):
        self.config = PlotConfig(**config) if isinstance(config, dict) else config or PlotConfig()
        self.data_path = data_path
        # self.fig: Optional[plt.Figure] = None
        # self.ax: Optional[plt.Axes] = None
        self.data: Any = None
        # 用户重写后填充这些
        self.x_vals: Optional[np.ndarray] = None
        self.y_vals: Optional[np.ndarray] = None
        self.subs: Optional[List[np.ndarray]] = None


    def new_fig(self, projection: str = 'rectilinear') -> None:
        """共性：创建fig/ax，支持polar（用户调用前指定）"""
        if projection == 'polar':
            self.fig, self.ax = plt.subplots(figsize=self.config.figsize, subplot_kw={'projection': 'polar'})
        else:
            self.fig, self.ax = plt.subplots(figsize=self.config.figsize)


    def re_initialized(self, config: Optional[Union[PlotConfig, Dict]] = None, data_path: Optional[str] = None) -> None:
        """共性：重置，支持链式调用"""
        self.__init__(config, data_path)
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
        # 示例：加载target，ref需手动
        self.data = load_lumerical_jsondata(self.data_path)

    def _load_pickle(self) -> None:
        """Pickle加载骨架（脚本2/3/4，用户重写后处理）"""
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        # 基础提取，用户在prepare_data重写扩展
        self.x_vals = self.data.get('x_vals', np.array([]))
        self.y_vals = self.data.get('y_vals', np.array([]))
        self.subs = self.data.get('subs', [])
        print(f"Pickle基础提取: x_shape={self.x_vals.shape}, subs_len={len(self.subs)} 🔍")

    @abstractmethod
    def prepare_data(self) -> None:
        """抽象：最灵活部分！用户手动重写：提取键、过滤NaN、计算衍生（e.g., PL_factor, 采样[::4]）"""
        pass

    @abstractmethod
    def plot(self) -> None:
        """抽象：用户手动重写绘图逻辑（e.g., 调用plot_line_advanced循环，或plot_2d_heatmap）"""
        pass


    def add_annotations(self) -> None:
        """共性：添加标签/限（用户可重写加自定义scale）"""
        if self.config.annotations:
            # 警示用户未设置
            print("Warning: 未设置标签")
        self.fig, self.ax = add_annotations(self.ax, self.config.annotations)
        plt.tight_layout()

    def add_twin_annotations(self) -> None:
        """共性：添加标签/限（用户可重写加自定义scale）"""
        if self.config.annotations:
            # 警示用户未设置
            print("Warning: 未设置标签")
        self.fig, self.twiny_ax = add_annotations(self.twiny_ax, self.config.annotations)
        plt.tight_layout()

    def save_and_show(self, custom_name: Optional[str] = None) -> None:
        """共性：保存/show（支持自定义名）"""
        full_params = self.config.plot_params or {}
        if custom_name:
            image_path = os.path.join(self.config.save_dir, custom_name)
        else:
            image_path = generate_save_name(self.config.save_dir, full_params)
        plt.savefig(image_path, dpi=self.config.dpi, bbox_inches="tight", transparent=True)
        print(f"图像已保存为：{image_path} 🎨")
        plt.savefig('temp_output.svg', dpi=self.config.dpi, bbox_inches="tight", transparent=True)
        print("Temp figure saved as 'temp_output.svg'.")
        if self.config.show:
            plt.show()

    def run_full(self) -> None:
        """可选完整链：load→prepare→new_fig→plot→add→save（用户若想一键）"""
        self.load_data()
        self.prepare_data()
        self.new_fig()
        self.plot()
        self.add_annotations()
        self.save_and_show()
        print("全流程完成！🚀")

# 分类子类1: LinePlotter（针对1D线类：脚本1的A/B/D + 脚本3的填充多线）
class LinePlotter(BasePlotter):
    """1D线类骨架：提供plot_line通用方法，用户重写prepare_data/plot调用它"""
    def plot_line(self, x: np.ndarray, z1: np.ndarray, twin=False, **kwargs) -> None:
        """辅助：通用plot_line_advanced（用户在plot中调用）"""
        params = {**self.config.plot_params, **kwargs}
        if twin:
            self.twiny_ax = self.ax.twiny()  # 共享 Y 轴
            self.twiny_ax = plot_line_advanced(self.twiny_ax, x, z1=z1, **params)
        else:
            self.ax = plot_line_advanced(self.ax, x, z1=z1, **params)

    # 用户在脚本重写prepare_data/plot，注入到此骨架

# 分类子类2: PolarPlotter（针对脚本1的C：极坐标）
class PolarPlotter(BasePlotter):
    """极坐标骨架：默认new_fig('polar')，提供plot_polar通用"""
    def plot_polar(self, theta: np.ndarray, radial: np.ndarray, **kwargs) -> None:
        """辅助：通用plot_polar_line"""
        params = {**self.config.plot_params, **kwargs}
        self.ax = plot_polar_line(self.ax, theta, radial, **params)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_thetalim(np.deg2rad(-60), np.deg2rad(60))  # 默认限，用户重写调整

# 分类子类3: HeatmapPlotter（针对脚本2的2D热图 + 脚本4的2D多线）
class HeatmapPlotter(BasePlotter):
    """2D类骨架：提供plot_heatmap/plot_multiline，用户重写Z准备"""
    def plot_heatmap(self, Z: np.ndarray, **kwargs) -> None:
        """辅助：plot_2d_heatmap"""
        params = {**self.config.plot_params, **kwargs}
        self.fig, self.ax = plot_2d_heatmap(self.ax, self.x_vals, self.y_vals, Z, params)

    def plot_multiline_2d(self, Z: np.ndarray, **kwargs) -> None:
        """辅助：plot_2d_multiline（支持alpha叠加）"""
        params = {**self.config.plot_params, **kwargs}
        self.fig, self.ax = plot_2d_multiline(self.ax, self.x_vals, self.y_vals, Z, params)
