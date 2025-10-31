from abc import ABC

from advance_plot_styles.polar_plot import *
from core.data_postprocess.momentum_space_toolkits import plot_isofreq_contours2D, extract_isofreq_paths, \
    sample_fields_along_path
from core.data_postprocess.polar_edges import plot_phi_families_split
from core.plot_workflow import *
from core.process_multi_dim_params_space import plot_advanced_surface
from utils.functions import skyrmion_density


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

    def plot_colored_bg(self) -> None:  # 重写：整体+循环填充
        params = {
            'enable_fill': True,
            'gradient_fill': True,
            # 'cmap': 'magma',
            'cmap': 'magma',
            'add_colorbar': False,
            'global_color_vmin': 0, 'global_color_vmax': 5e-3,
            'default_color': 'gray', 'alpha_fill': 1,
            'edge_color': 'none',
            'gradient_direction': 'z3',
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            self.plot_line(x, z1=y.real, z2=y.imag, z3=y.imag, **params)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.ax.set_xlim(self.x_vals.min(), self.x_vals.max())
        self.ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)

    def plot_colored_line(self) -> None:  # 重写：整体+循环填充
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
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line)  # 填充


class MomentumSpaceEigenPolarizationPlotter(BasePlotter, ABC):
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

    def plot_polarization_ellipses(self, index, step=(5, 5)) -> None:
        self.ax = plot_polarization_ellipses(
            self.ax, self.Mx, self.My, self.s1_list[index], self.s2_list[index], self.s3_list[index],
            step=step,  # 适当抽样，防止太密
            scale=1e-2,  # 自动用 0.8*min(dx,dy)
            cmap='RdBu',
            alpha=1, lw=1,
        )
        # 重新设置画布的视角
        self.ax.set_xlim(self.Mx.min(), self.Mx.max())
        self.ax.set_ylim(self.My.min(), self.My.max())

    def plot_isofreq_contours2D(self, index, levels=(0.509, 0.510, 0.511)) -> None:
        self.ax = plot_isofreq_contours2D(
            self.ax, self.m1, self.m2, self.eigenfreq_list[index].T, levels=levels,
            colors=['k', 'k', 'k'],
            linewidths=1.0
        )

    def plot_phi_families_regimes(self, index) -> None:
        color_1 = 'lightgreen'
        color_2 = 'lightcoral'
        # 绘制区域
        self.ax.contourf(self.m1, self.m1, (np.sin(2 * self.phi_list[index].T) > 0), levels=[-0.5, 0.5],
                         colors=[color_2, color_1], alpha=0.5)

    def plot_phi_families_split(self, index) -> None:
        self.ax = plot_phi_families_split(self.ax, self.m1, self.m2, self.phi_list[index], overlay=None, lw=1)

    def sample_along_isofreq(self, index=0, level=0.510) -> None:
        m1f, m2f = self.m1, self.m2
        phi_f, tanchi_f = self.phi_list[index], self.tanchi_list[index]
        Z_f = self.eigenfreq_list[index]
        qlog_f = np.log10(np.where(Z_f.imag != 0, np.abs(Z_f.real / (2 * Z_f.imag)), 1e10))
        paths = extract_isofreq_paths(m1f, m1f, Z_f, level=level)
        # 沿每条等频线插值采样
        fields = {'phi': phi_f, 'chi': tanchi_f, 'Q': qlog_f, 'freq': Z_f.real}
        samples_list = []
        for p in paths:
            samp = sample_fields_along_path(m1f, m2f, fields, p, npts=400)
            samples_list.append(samp)
        # 画曲线
        fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
        ax.plot(samp['s'], samp['phi'])
        fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
        ax.plot(samp['s'], samp['chi'])
        fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
        ax.plot(samp['s'], samp['Q'])
        plt.show()

    def imshow_qlog(self, index=0) -> None:
        self.ax.imshow(self.qlog_list[index].T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', cmap='hot', aspect='equal')

    def imshow_phi(self, index=0) -> None:
        self.ax.imshow(self.phi_list[index].T, extent=(self.m1.min(), self.m1.max(), self.m2.min(), self.m2.max()),
                       origin='lower', cmap='twilight',
                       aspect='equal')

    def plot_3D_surface(self, index, mapping=None) -> None:
        if mapping is None:
            mapping = {
                'cmap': 'hot',
                # 'z2': {'vmin': a, 'vmax': b},  # 可选；未给则自动取数据范围
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
            elev=30, azim=25,
            font_size=9,
        )

    def plot_skyrmion_quiver(self, index):
        self.ax = plot_skyrmion_quiver(
            self.ax, self.Mx, self.My,
            self.s1_list[index], self.s2_list[index], self.s3_list[index],
            step=(6, 6),
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
        return nsk


