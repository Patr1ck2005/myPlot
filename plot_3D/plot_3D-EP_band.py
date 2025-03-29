import pandas as pd

from core.plot_3D_params_space import plot_3D_params_space

if __name__ == '__main__':

    # 载入数据并调用函数
    # data_path = './data/EP_band-tgrating_lowloss.csv'
    # data = pd.read_csv(data_path, sep='\t').to_numpy()
    # plot_3D_params_space(
    #     data,
    #     save_label=data_path.split('/')[-1].split('.')[0],
    #     selected_params={
    #         'k1': 0.02,
    #         'Qlim': (0, 100)
    #     },
    #     zlim=[0, 0.3],
    # )

    # 载入数据并调用函数
    data_path = './data/EP_band-tgrating_details.csv'
    data = pd.read_csv(data_path, sep='\t').to_numpy()
    plot_3D_params_space(
        data,
        save_label=data_path.split('/')[-1].split('.')[0],
        selected_params={
            'k1': 0.08,
            'Qlim': (0, 30)
        },
        zlim=[0.4, 1.5],
        norm_zlim=[0.4, 1.0],
        cmap='twilight',
    )