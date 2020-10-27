import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_ucsf = pd.read_csv('raw_data/ucsf_volpara.csv', low_memory=False)
df_ucsf['ImageName'] = ['ucsf_' + x for x in df_ucsf['ImageName'].astype(str).values]
df_mayo = pd.read_csv('raw_data/mayo_volpara.csv', low_memory=False)
df_mayo['ImageName'] = ['mayo_' + x for x in df_mayo['ImageName'].astype(str).values]

df = pd.concat([df_ucsf, df_mayo])

df_out = pd.DataFrame()
df_out['dicom_id'] = df['ImageName']
df_out['volumetric_breast_density'] = df['VolumetricBreastDensity']
df_out = df_out.dropna()
df_out = df_out.sort_values('volumetric_breast_density')
df_out.to_csv('out_data/volpara.csv', index=False)

# plt.figure(figsize=(10, 10))
# sns.distplot(df_ucsf['VolumetricBreastDensity'].dropna(), label='ucsf')
# sns.distplot(df_mayo['VolumetricBreastDensity'].dropna(), label='mayo')
# plt.savefig('a.png')
