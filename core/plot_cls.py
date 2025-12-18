from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from advance_plot_styles.polar_plot import *
from core.data_postprocess.momentum_space_toolkits import plot_isofreq_contours2D, extract_isofreq_paths, \
    sample_fields_along_path
from core.data_postprocess.polar_graph_analysis import plot_phi_families_split
from core.plot_workflow import *
from core.process_multi_dim_params_space import plot_advanced_surface
from utils.advanced_color_mapping import map_s1s2s3_color, map_complex2rbg
from utils.functions import skyrmion_density, skyrmion_number, lorenz_func


class BandPlotterOneDim(LinePlotter, ABC):
    """能带图骨架"""

    def prepare_data(self, momen_key='k') -> None:  # 手动重写：NaN过滤
        self.x_vals_list = []
        self.y_vals_list = []
        for raw_data in self.raw_datasets["data_list"]:
            sub = raw_data['eigenfreq']
            mask = np.isnan(sub)
            self.x_vals = self.coordinates[momen_key]
            if np.any(mask):
                print("Warning: NaN移除⚠️")
                self.y_vals = sub[~mask]
                temp_x = self.x_vals[~mask]
            else:
                self.y_vals = sub
                temp_x = self.x_vals
            self.x_vals_list.append(temp_x)
            self.y_vals_list.append(self.y_vals)

    def plot(self, style='1') -> None:  # 重写：整体+循环填充
        if style == '1':
            params_bg = {
                'enable_fill': True,
                'gradient_fill': False,
                'enable_dynamic_color': False,
                'cmap': None,
                'add_colorbar': False,
                'global_color_vmin': 1, 'global_color_vmax': 8,
                'default_color': 'gray', 'alpha_fill': 0.3,
                'edge_color': 'none',
                'gradient_direction': 'z3',
                'linewidth_base': 0,
            }
            params_line = {
                'enable_fill': False,
                'gradient_fill': False,
                'enable_dynamic_color': True,
                'cmap': 'hot',
                'add_colorbar': False,
                'global_color_vmin': 1, 'global_color_vmax': 8,
                'default_color': 'gray', 'alpha_fill': 1,
                'linewidth_base': 2,
                'edge_color': 'none',
            }
            y_mins, y_maxs = [], []
            for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
                Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
                Qfactor_log = np.log10(Qfactor)
                self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_bg)  # 填充
                self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line)  # 填充
                widths = np.abs(y.imag)
                y_mins.append(np.min(y.real - widths))
                y_maxs.append(np.max(y.real + widths))
            self.ax.set_xlim(self.x_vals.min(), self.x_vals.max())
            self.ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)

    def plot_thick_bg(self) -> None:  # 重写：整体+循环填充
        params_bg = {
            'enable_fill': True,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': None,
            'add_colorbar': False,
            'global_color_vmin': 1, 'global_color_vmax': 8,
            'default_color': 'gray', 'alpha_fill': 0.3,
            'edge_color': 'none',
            'gradient_direction': 'z3',
            'linewidth_base': 0,
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_bg)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)

    def plot_colored_bg(self, vmin=0, vmax=5e-3, alpha=1, y_margin=0.02) -> None:  # 重写：整体+循环填充
        params = {
            'enable_fill': True,
            'gradient_fill': True,
            # 'cmap': 'magma',
            'cmap': 'magma',
            'add_colorbar': False,
            'global_color_vmin': vmin, 'global_color_vmax': vmax,
            'default_color': 'gray', 'alpha_fill': alpha,
            'edge_color': 'none',
            'gradient_direction': 'z3',
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            self.plot_line(x, z1=y.real, z2=y.imag, z3=y.imag, **params)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))

    def plot_colored_line(self, vmin=2, vmax=7, cmap='magma', y_margin=0.02) -> None:  # 重写：整体+循环填充
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': True,
            'cmap': cmap,
            'add_colorbar': False,
            'global_color_vmin': vmin, 'global_color_vmax': vmax,
            'default_color': 'gray', 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
            'alpha_line': 1
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line)  # 填充
            y_mins.append(np.min(y.real))
            y_maxs.append(np.max(y.real))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))

    def plot_ordered_line(self, y_margin=0.02) -> None:  # 重写：整体+循环填充
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': None,
            'add_colorbar': False,
            'default_color': False, 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
            'alpha_line': 0.75,
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line, index=i)  # 填充
            y_mins.append(np.min(y.real))
            y_maxs.append(np.max(y.real))
        self.xlim = (self.x_vals.min(), self.x_vals.max())
        self.ylim = (np.nanmin(y_mins) * (1-y_margin), np.nanmax(y_maxs) * (1+y_margin))
        self.ax.legend()


    def plot_diffraction_cone(self, env_n=1, scale=1, upper_limit=1, color='lightgreen', alpha=0.2) -> None:
        # 绘制衍射锥线
        kx = self.x_vals
        c = 1  # 光速归一化
        diffraction_cone = (1 - c * np.abs(kx)) / env_n
        # 填充衍射极限以上的区域
        self.ax.fill_between(kx, diffraction_cone*scale, upper_limit, color=color, alpha=alpha, edgecolor='none', zorder=0)



class MomentumSpaceEigenPolarizationPlotter(HeatmapPlotter, ABC):
    """极化图骨架"""

    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        self.data_list = self.raw_datasets["data_list"]
        self.eigenfreq_list = [self.data_list[i]['eigenfreq'] for i in range(len(self.data_list))]
        self.s1_list = [self.data_list[i]['s1'] for i in range(self.data_num)]
        self.s2_list = [self.data_list[i]['s2'] for i in range(self.data_num)]
        self.s3_list = [self.data_list[i]['s3'] for i in range(self.data_num)]
        self.qlog_list = [self.data_list[i]['qlog'] for i in range(self.data_num)]

    def interpolate_data(self, factor=2) -> None:
        # 对所有数据进行插值
        from scipy.ndimage import zoom
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        new_m1 = np.linspace(self.m1.min(), self.m1.max(), len(self.m1) * factor)
        new_m2 = np.linspace(self.m2.min(), self.m2.max(), len(self.m2) * factor)
        self.m1 = new_m1
        self.m2 = new_m2
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')

        def interp_field(field):
            return zoom(field, zoom=factor, order=3)

        self.eigenfreq_list = [interp_field(self.data_list[i]['eigenfreq']) for i in range(len(self.data_list))]
        self.s1_list = [interp_field(self.data_list[i]['s1']) for i in range(self.data_num)]
        self.s2_list = [interp_field(self.data_list[i]['s2']) for i in range(self.data_num)]
        self.s3_list = [interp_field(self.data_list[i]['s3']) for i in range(self.data_num)]
        self.qlog_list = [interp_field(self.data_list[i]['qlog']) for i in range(self.data_num)]

    def plot(self) -> None:
        self.new_2d_fig()
        self.plot_polarization_ellipses(index=0)
        plt.show()

        self.new_3d_fig()
        self.plot_3D_surface(index=0)
        plt.show()

        self.new_2d_fig()
        self.plot_phi_families_regimes(index=0)
        plt.show()

        self.new_2d_fig()
        self.plot_phi_families_split(index=0)
        plt.show()

    def plot_skyrmion_analysis(self, index) -> None:
        self.prepare_chi_phi_data()

        self.new_3d_fig(temp_figsize=(3, 3))
        self.plot_on_poincare_sphere(index)
        plt.show()

        self.new_2d_fig()
        self.plot_polarization_ellipses(index)
        plt.show()

        self.new_2d_fig()
        self.plot_phi_families_regimes(index)
        self.plot_phi_families_split(index)
        plt.show()

        self.new_2d_fig()
        self.imshow_advanced_color_mapping(index)
        plt.show()

        self.new_2d_fig()
        nsk = self.imshow_skyrmion_density(index)
        plt.show()

    def prepare_chi_phi_data(self) -> None:
        # 通过s123计算phi, tanchi
        self.phi_list = []
        self.tanchi_list = []
        for i in range(self.data_num):
            s1 = self.s1_list[i]
            s2 = self.s2_list[i]
            s3 = self.s3_list[i]
            # 计算 |cos(2χ)|，在 χ ∈ [-π/4, π/4] 的主值范围内 cos(2χ)≥0，因此取正根即可
            cos2chi_abs = np.sqrt(np.maximum(s1 ** 2 + s2 ** 2, 0.0))
            # 2χ = atan2(sin2χ, cos2χ)，其中 cos2χ≥0（主值保证）
            chi = 0.5 * np.arctan2(s3, cos2chi_abs)
            # phi 的 2φ = atan2(s2, s1)，自动给出正确象限；纯圆时 s1=s2=0，phi 无定义
            phi = (0.5 * np.arctan2(s2, s1)) % np.pi
            self.phi_list.append(phi % np.pi)
            self.tanchi_list.append(np.tan(chi))

    def plot_polarization_ellipses(self, index, step=(1, 1), scale=1e-2) -> None:
        self.ax = plot_polarization_ellipses(
            self.ax, self.Mx, self.My, self.s1_list[index], self.s2_list[index], self.s3_list[index],
            step=step,  # 适当抽样，防止太密
            scale=scale,  # 自动用 0.8*min(dx,dy)
            cmap='RdBu',
            alpha=1, lw=1,
        )
        # 重新设置画布的视角
        self.ax.set_xlim(self.Mx.min(), self.Mx.max())
        self.ax.set_ylim(self.My.min(), self.My.max())

    def get_advanced_color_mapping(self, index) -> np.ndarray:
        rgb = map_s1s2s3_color(self.s1_list[index], self.s2_list[index], self.s3_list[index])
        return rgb

    def imshow_advanced_color_mapping(self, index) -> None:
        rgb = self.get_advanced_color_mapping(index)
        self.ax.imshow(np.transpose(rgb, (1, 0, 2)),
                       extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal')

    def plot_isofreq_contours2D(self, index, levels=(0.509, 0.510, 0.511), colors=('k', 'k', 'k'), cmap=None) -> None:
        if cmap is not None:
            # 使用 colormap 生成颜色列表
            from matplotlib import cm
            colormap = cm.get_cmap(cmap, len(levels))
            colors = [colormap(i) for i in range(len(levels))]
        self.ax = plot_isofreq_contours2D(
            self.ax, self.m1, self.m2, self.eigenfreq_list[index].T, levels=levels,
            colors=colors,
            linewidths=1.0
        )

    def plot_phi_families_regimes(self, index) -> None:
        # color_1 = 'lightgreen'
        color_1 = 'white'
        color_2 = 'lightcoral'
        # 绘制区域
        self.ax.contourf(self.m1, self.m2, (np.sin(2 * self.phi_list[index].T) > 0), levels=[-0.5, 0.5, 1.5],
                         colors=[color_2, color_1], alpha=0.5)

    def plot_phi_families_split(self, index) -> None:
        self.ax = plot_phi_families_split(self.ax, self.m1, self.m2, self.phi_list[index], overlay=None, lw=1)

    def sample_along_isofreq(self, index=0, level=None, show=False):
        assert level is not None, "请提供等频线频率值 level"
        m1f, m2f = self.m1, self.m2
        phi_f, tanchi_f = self.phi_list[index], self.tanchi_list[index]
        Z_f = self.eigenfreq_list[index]
        qlog_f = np.log10(np.where(Z_f.imag != 0, np.abs(Z_f.real / (2 * Z_f.imag)), 1e10))
        paths = extract_isofreq_paths(m1f, m1f, Z_f, level=level)
        # 沿每条等频线插值采样
        fields = {'phi': phi_f, 'tanchi': tanchi_f, 'qlog': qlog_f, 'freq': Z_f.real}
        samples_list = []
        for p in paths:
            samp = sample_fields_along_path(m1f, m2f, fields, p, npts=400)
            samples_list.append(samp)
        if show:
            # 画曲线
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['phi'])
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['tanchi'])
            fig, ax = plt.subplots(figsize=(1.25, 1.25))
            ax.plot(samp['s'], samp['qlog'])
            plt.show()
        return samples_list

    def imshow_qlog(self, index=0, **kwargs) -> None:
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'hot'
        self.ax.imshow(self.qlog_list[index].T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal', **kwargs)

    def imshow_phi(self, index=0, **kwargs) -> None:
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'twilight'
        self.ax.imshow(self.phi_list[index].T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower',
                       aspect='equal', **kwargs)

    def imshow_s3(self, index=0) -> None:
        self.ax.imshow(self.s3_list[index].T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', cmap='RdBu',
                       aspect='equal')

    def plot_3D_surface(self, index, mapping=None, rbga=None, shade=True, elev=45, azim=25, **kwargs) -> None:
        if mapping is None:
            mapping = {
                'cmap': 'hot',
                'z2': {'vmin': 2, 'vmax': 7},  # 可选；未给则自动取数据范围
                # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
            }
        m1f, m2f = self.m1, self.m2
        eigenfreq = self.eigenfreq_list[index]
        qlog = self.qlog_list[index]
        self.ax, mappable = plot_advanced_surface(
            self.ax, mx=m1f, my=m2f,
            mapping=mapping,
            z1=eigenfreq,
            z2=qlog,
            rbga=rbga,
            elev=elev, azim=azim, shade=shade,
            **kwargs
        )

    def plot_skyrmion_quiver(self, index, step=(6, 6)) -> None:
        self.ax = plot_skyrmion_quiver(
            self.ax, self.Mx, self.My,
            self.s1_list[index], self.s2_list[index], self.s3_list[index],
            step=step,
            normalize=True,
            cmap='RdBu',
            clim=(-1, 1),
            quiver_scale=None,
            width=0.006,
        )

    def plot_on_poincare_sphere(self, index):
        self.ax = plot_on_poincare_sphere(
            self.ax,
            self.s1_list[index], self.s2_list[index], self.s3_list[index],
            S0=None,
            step=(1, 1),
            c_by='s3',
            cmap='RdBu',
            clim=(-1, 1),
            s=8,
            alpha=0.9,
            sphere_style='wire',
        )

    def imshow_skyrmion_density(self, index):
        nsk = skyrmion_density(
            self.s1_list[index], self.s2_list[index], self.s3_list[index]
        )
        self.ax.imshow(nsk.T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', cmap='bwr', aspect='equal')

        print(f"full-plane")
        s = skyrmion_number(nsk, show=False)
        left_half_mask = (self.Mx < 0)
        print(f"left half-plane")
        s_left = skyrmion_number(nsk, mask=left_half_mask, show=False)
        left_round_mask = (self.Mx ** 2 + self.My ** 2 < 0.05 ** 2) * (self.Mx < 0)
        print(f"left half-plane")
        s_left_round = skyrmion_number(nsk, mask=left_round_mask, show=False)
        return nsk

    def compute_cross_polarization_conversion(self, index=0, freq=None):
        cross_conversion = np.exp(-2j*self.phi_list[index])*lorenz_func(delta_omega=(freq-self.eigenfreq_list[index].real), gamma=self.eigenfreq_list[index].imag, gamma_nr=0)
        self.cross_conversion = cross_conversion
        # self.ax.imshow(np.real(cross_conversion).T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
        #                   origin='lower', cmap='RdBu', aspect='equal')


class MomentumSpaceSpectrumPlotter(HeatmapPlotter, ABC):
    """光谱图骨架"""
    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.m1 = self.coordinates['m1']
        self.m2 = self.coordinates['m2']
        self.Mx, self.My = np.meshgrid(self.m1, self.m2, indexing='ij')
        self.data_list = self.raw_datasets["data_list"]
        self.s11_list = [self.data_list[i]['s11'] for i in range(self.data_num)]
        self.s21_list = [self.data_list[i]['s21'] for i in range(self.data_num)]

    def prepare_helical_basis(self) -> None:
        pass

    def plot(self) -> None:
        pass

    def plot_skyrmion_analysis(self, index) -> None:
        pass

    def get_advanced_color_mapping(self, index) -> np.ndarray:
        rgb = map_s1s2s3_color(self.s1_list[index], self.s2_list[index], self.s3_list[index])
        return rgb

    def imshow_advanced_color_mapping(self, index) -> None:
        rgb = self.get_advanced_color_mapping(index)
        self.ax.imshow(np.transpose(rgb, (1, 0, 2)),
                       extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal')

    def imshow_s11(self, index=0, **kwargs) -> None:
        # if 'cmap' not in kwargs:
        #     kwargs['cmap'] = 'RdBu'
        rgb = map_complex2rbg(self.s11_list[index])
        self.ax.imshow(rgb, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal', **kwargs)

    def imshow_s21(self, index=0, **kwargs) -> None:
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu'
        rgb = map_complex2rbg(self.s21_list[index])
        self.ax.imshow(rgb, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', aspect='equal', **kwargs)


class TwoDimFieldVisualizer:
    """
    通用二维参数空间可视化器
    - 不理解物理意义
    - 只管理坐标、字段和绘制策略
    """

    def __init__(self, x, y, fields: dict):
        """
        x, y: 1D arrays
        fields: dict[str, ndarray], shape=(Nx, Ny) or (Nx, Ny, ...)
        """
        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.fields = fields

    # ---------- 字段变换 ----------

    def add_field(self, name, value):
        self.fields[name] = value

    def compute_phi_tanchi_from_s123(self, s1_key='s1', s2_key='s2', s3_key='s3'):
        s1 = self.fields[s1_key]
        s2 = self.fields[s2_key]
        s3 = self.fields[s3_key]

        cos2chi = np.sqrt(np.maximum(s1**2 + s2**2, 0))
        chi = 0.5 * np.arctan2(s3, cos2chi)
        phi = (0.5 * np.arctan2(s2, s1)) % np.pi

        self.fields['phi'] = phi
        self.fields['tanchi'] = np.tan(chi)

    def compute_qlog_from_eigenfreq(self, key='eigenfreq'):
        Z = self.fields[key]
        qlog = np.log10(np.where(Z.imag != 0, np.abs(Z.real / (2 * Z.imag)), np.nan))
        self.fields['qlog'] = qlog

    # ---------- 绘制入口 ----------

    def plot(self, strategy, field, **kwargs):
        fig, ax = plt.subplots()
        strategy(ax, self, field, **kwargs)
        return fig, ax


