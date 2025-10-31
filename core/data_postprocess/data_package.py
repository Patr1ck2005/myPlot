from typing import Dict, Sequence

import numpy as np

from core.data_postprocess.momentum_space_toolkits import complete_C2_polarization, geom_complete, \
    complete_C4_polarization
from core.process_multi_dim_params_space import extract_basic_analysis_fields
from utils.functions import ellipse2stokes

GridCoords = Dict[str, np.ndarray]
ZArray = np.ndarray  # object 数组（你的 Z / Z_target 结构）

def package_stad_C2_data(
        coords: GridCoords,
        band_index: int,
        main_data: ZArray,
        additional_data: ZArray,
        z_keys: Sequence[str],
        **kwargs
):
    # 提取 band= 的附加场数据
    phi, tanchi, qlog, freq_real = extract_basic_analysis_fields(
        additional_data, z_keys=z_keys, band_index=band_index, **kwargs
    )
    full_coords, phi_f, tanchi_f = complete_C2_polarization(coords, phi, tanchi)
    _, main_data = geom_complete(coords, main_data, mode='x')
    _, qlog_f = geom_complete(coords, qlog, mode='x')
    s1, s2, s3 = ellipse2stokes(phi_f, tanchi_f)
    dataset = {
        'eigenfreq': main_data,
        's1': s1,
        's2': s2,
        's3': s3,
        'qlog': qlog_f,
    }
    return full_coords, dataset


def package_stad_C4_data(
        coords: GridCoords,
        band_index: int,
        main_data: ZArray,
        additional_data: ZArray,
        z_keys: Sequence[str],
        **kwargs
):
    # 提取 band= 的附加场数据
    phi, tanchi, qlog, freq_real = extract_basic_analysis_fields(
        additional_data, z_keys=z_keys, band_index=band_index, **kwargs
    )
    full_coords, phi_f, tanchi_f = complete_C4_polarization(coords, phi, tanchi)
    _, main_data = geom_complete(coords, main_data, mode='xy')
    _, qlog_f = geom_complete(coords, qlog, mode='xy')
    s1, s2, s3 = ellipse2stokes(phi_f, tanchi_f)
    dataset = {
        'eigenfreq': main_data,
        's1': s1,
        's2': s2,
        's3': s3,
        'qlog': qlog_f,
    }
    return full_coords, dataset