import pandas as pd

from plot_3D.core.plot_3D_params_space import plot_3D_params_space

if __name__ == '__main__':

    # 载入数据并调用函数
    data_path = './data/EP_flatband.csv'
    data = pd.read_csv(data_path, sep='\t').to_numpy()
    plot_3D_params_space(
        data,
        save_label=data_path.split('/')[-1].split('.')[0],
        selected_params={
            'k1': 0.0,
            'Qlim': (200, -1),
            'freq_lim': (60, 70),
            # 'z_type': 'freq_real',
            'z_type': 'freq_im',
        },
        # zlim=[0.125, 0.25],
        norm_zlim=[0, 0.15],  # for imag
        # norm_zlim=[62.8, 63.2],  # for real
        cmap='twilight',
    )