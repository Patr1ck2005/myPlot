from abc import ABC

from core.plot_workflow import *


class BandPlotterOneDim(LinePlotter, ABC):
    """能带图骨架"""
    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.x_vals_list = []
        self.y_vals_list = []
        for raw_data in self.raw_datasets["data_list"]:
            sub = raw_data['eigenfreq']
            mask = np.isnan(sub)
            self.x_vals = self.coordinates['k']
            if np.any(mask):
                print("Warning: NaN移除⚠️")
                self.y_vals = sub[~mask]
                temp_x = self.x_vals[~mask]
            else:
                self.y_vals = sub
                temp_x = self.x_vals
            self.x_vals_list.append(temp_x)
            self.y_vals_list.append(self.y_vals)


class PolarizationPlotter(BasePlotter, ABC):
    """极化图骨架"""
    pass

