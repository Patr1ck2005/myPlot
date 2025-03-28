import pandas as pd

from core.plot_3D_params_space import plot_3D_params_space

if __name__ == '__main__':

    # 载入数据并调用函数
    data_path = './data/EP_band-tgrating_lowloss.csv'
    data = pd.read_csv(data_path, sep='\t').to_numpy()
    plot_3D_params_space(
        data,
        save_label=data_path.split('/')[-1].split('.')[0]
    )