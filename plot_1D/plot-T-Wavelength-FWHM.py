from plot_1D.plot_dual_y_axis import plot_two_scales

temperature = [175, 200, 225, 250, 275]  # X 轴: 温度
wavelength = [6.386, 6.406, 6.436, 6.458, 6.484]  # 左 Y 轴: 波长
fwhm = [26, 23, 32, 35, 46]  # 右 Y 轴: FWHM

# 调用通用函数
plot_two_scales(
    temperature, wavelength, fwhm,
    x_label='Temperature (°C)',
    x_ticks=[175, 200, 225, 250, 275, 300],
    y1_label='Center Wavelength (µm)',
    y1_lim=[6.35, 6.5],
    y2_label='FWHM (nm)',
    y1_color='brown',
    y2_color='navy',
    title='',
    y1_marker='o',
    y2_marker='s',
    save_name='iso_ThE',
    linestyle='--',
    # marker_size=3
)