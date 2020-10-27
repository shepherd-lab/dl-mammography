from xz import display_imgs, chunk_df
import os
from glob import glob
import numpy as np
import pandas as pd

base_dir = 'out_tmp'

pmeta = pd.read_csv(f"{base_dir}/pmeta.csv", index_col=0)
dmeta = pd.read_csv(f"{base_dir}/dmeta.csv", index_col=0)

# TODO: make sure that the four panels are sorted consistently
# TODO: investigate the missing files at the end of the table

output_dir = f"{base_dir}/contact_sheets_png"
os.makedirs(output_dir, exist_ok=True)
for i_chunk, pmeta_chunk in enumerate(chunk_df(pmeta, 60)):
    img_list = []
    for patient_id, row in pmeta_chunk.iterrows():
        sub_img_list = []
        sub_img_df = dmeta.loc[dmeta['patient_id'] == patient_id]
        for dicom_id, row2 in sub_img_df.iterrows():
            sub_img_list.append(np.load(f"{base_dir}/npys_for_cnn/{dicom_id}.npy"))
        sub_img_labels = ', '.join(
            f"{a}_{b}" for a, b in zip(sub_img_df['image_laterality'], sub_img_df['view_position'])
        )
        label = '\n'.join([
            patient_id,
            *[f"{k} = {v}" for k, v in row.iteritems()],
            sub_img_labels,
        ])
        img_list.append((sub_img_list, label))
    display_imgs(img_list, save_as=f"{output_dir}/chunk_{i_chunk:0>2}.png", aspect_ratio=1.75, dpi=50, n_cols=14)
