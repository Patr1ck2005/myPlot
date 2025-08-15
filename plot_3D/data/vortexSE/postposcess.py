import numpy as np
import pandas as pd

data_path = 'Hex_hole-center-0.4r_0.1t-0.05kr-freqs-pre.csv'
df_sample = pd.read_csv(data_path, sep='\t')
cl = df_sample['cx+i*cy (V/m)'].apply(lambda x: complex(x.replace('i', 'j')))
cr = df_sample['cx-i*cy (V/m)'].apply(lambda x: complex(x.replace('i', 'j')))

cx = (cl+cr)/2
cy = (cl-cr)/2j

S0 = np.abs(cx)**2+np.abs(cy)**2
S1 = (np.abs(cx)**2-np.abs(cy)**2)/S0
S2 = 2*np.real(np.conj(cx)*cy)/S0
S3 = 2*np.imag(np.conj(cx)*cy)/S0

df_sample['abs(cx)^2+abs(cy)^2 (kg^2*m^2/(s^6*A^2))'] = S0
df_sample['(abs(cx)^2-abs(cy)^2)/(abs(cx)^2+abs(cy)^2) (1)'] = S1
df_sample['2*real(conj(cx)*cy)/(abs(cx)^2+abs(cy)^2) (1)'] = S2
df_sample['2*imag(conj(cx)*cy)/(abs(cx)^2+abs(cy)^2) (1)'] = S3

df_sample.to_csv('Hex_hole-center-0.4r_0.1t-0.05kr-freqs.csv', sep='\t', index=False)

