# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np

from core.plot_workflow import PlotConfig, HeatmapPlotter
from core.plot_cls import BandPlotterOneDim


class MyScriptPlotter(HeatmapPlotter):
    def load_data(self) -> None:
        data_path = f'A_spectrum_[45776677761F3053425D]/RTA_spectrum-{v_perc}%V-7-7-7-PMMA_bg__A_spectrum.csv'
        self.spectrum_data2d = np.loadtxt(self.data_path+data_path, delimiter=',', skiprows=1)
        data_path = f'angles_[6666671E7C53]/RTA_spectrum-{v_perc}%V-7-7-7-PMMA_bg__angles.csv'
        self.para1_vals = np.loadtxt(self.data_path+data_path, delimiter=',', skiprows=1)
        data_path = f'wavelengths_[767666667677165C5E7483]/RTA_spectrum-{v_perc}%V-7-7-7-PMMA_bg__wavelengths.csv'
        self.para2_vals = np.loadtxt(self.data_path+data_path, delimiter=',', skiprows=1)

    def prepare_data(self) -> None:
        self.load_data()

    def plot(self) -> None:
        self.ax.imshow(
            self.spectrum_data2d.T,
            extent=[
                np.min(self.para1_vals),
                np.max(self.para1_vals),
                np.min(self.para2_vals)*1e6,
                np.max(self.para2_vals)*1e6,
            ],
            aspect='auto',
            origin='lower',
            # cmap='magma',
            cmap='rainbow',
            interpolation='none',
            vmin=0, vmax=1
        )
        # self.ax.set_title('A Spectrum Heatmap')
        # add colorbar
        self.ax.figure.colorbar(self.ax.images[0], ax=self.ax, label='Absorption')




def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={'xlabel': 'Angle (°)', 'ylabel': 'Wavelength ($\mu$m)', 'show_axis_labels': True, 'show_tick_labels': True},
    )
    config.figsize = (2, 3)
    config.tick_direction = 'in'
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot()
    plotter.adjust_view_2dim()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    v_perc = 10
    data_root = f'data/Nanoparticals-FDTD_size_7_7_7um-#%V-PMMA_bg-MIR_8_14um/RTA_spectrum-{v_perc}%V-7-7-7-PMMA_bg_csv/'
    main(data_path=data_root)
