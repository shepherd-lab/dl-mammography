import math
from scipy import signal
import sys
import re
import sys
import scipy.fftpack
import os
from collections import OrderedDict
import cv2

from scipy.ndimage import filters
import fire
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from xz import chunk_df, debug, display_imgs, parallelize, warn

# TODO: set the pixel inverting as a parameter

config = {
    'output_base_dir': 'out_data/pres512_2',
    'pixel_max_value': (2**14) - 1,
    'crop_width': 2500,
    'crop_height': 3000,
    'cnn_input_width': 512,
    'cnn_input_height': 512,
    'n_samples': None,
    'blur_radius': 64,
    'plot_dpi': 100,

    # FOR RAW
    # 'basedir_to_mayo_dicoms': 'raw_data/mayo_dicoms',
    # 'basedir_to_ucsf_dicoms': 'raw_data/ucsf_dicoms',
    # 'filename_suffix': '',
    # 'dense_dark': True,

    # FOR PRES
    'basedir_to_mayo_dicoms': 'raw_data/mayo_dicoms_pres',
    'basedir_to_ucsf_dicoms': 'raw_data/ucsf_dicoms_pres',
    'filename_suffix': '_pres',
    'dense_dark': False,
}

os.makedirs(f"{config['output_base_dir']}/npys", exist_ok=True)
os.makedirs(f"{config['output_base_dir']}/npys_for_cnn", exist_ok=True)
os.makedirs(f"{config['output_base_dir']}/excluded", exist_ok=True)


def process_ucsf_pmeta(path_to_df):
    df = pd.read_csv(path_to_df, dtype={'linkage_id': str, 'acquisition_id': str})
    df = df[['linkage_id', 'DETECT']].drop_duplicates()

    dff = pd.DataFrame()
    dff['patient_id'] = 'ucsf_' + df['linkage_id']
    dff['label'] = df['DETECT']
    return dff


def process_ucsf_dmeta(path_to_df):
    df = pd.read_csv(path_to_df, dtype={'linkage_id': str, 'acquisition_id': str})
    dff = pd.DataFrame()
    dff['patient_id'] = 'ucsf_' + df['linkage_id']
    dff['dicom_id'] = 'ucsf_' + df['acquisition_id']

    dff['path_to_dicom'] = [
        f"config['basedir_to_ucsf_dicoms']/{id_}{config['filename_suffix']}.dcm" for id_ in df['acquisition_id']
    ]

    return dff


def process_mayo_pmeta(path_to_df):
    df = pd.read_csv(path_to_df)
    dff = pd.DataFrame()
    dff['patient_id'] = 'mayo_' + df['encrypted']
    dff['label'] = df['status']
    return dff


def process_mayo_dmeta(path_to_df):
    df = pd.read_csv(path_to_df)
    df = df.melt(id_vars=['encrypted', 'status'])
    df = df.dropna()
    df['value'] = [x[:-4] for x in df['value']]  # remove the ".dcm" at the end

    dff = pd.DataFrame()
    dff['patient_id'] = 'mayo_' + df['encrypted']
    dff['dicom_id'] = 'mayo_' + df['value']

    dff['path_to_dicom'] = [f"{config['basedir_to_mayo_dicoms']}/{id_}{config['filename_suffix']}.dcm" for id_ in df['value']]

    dff = dff.sort_values(['patient_id', 'dicom_id'])

    return dff


def crop_one_image_for_geras(fp):
    try:
        img = np.load(fp)
        basename = os.path.basename(fp)
        img = process_an_img_for_cnn(img)
        np.save(f"out_data/npys_for_geras/{basename}", img)
        return {'id_': f"{fp}  -->  out_data/npys_for_geras/{basename}"}
    except Exception as e:
        return {'id_': f"{fp} ERROR: {e}", 'errors': [e]}


def process_an_img_for_cnn(img):
    height, width = img.shape

    if config['crop_height'] is None or config['crop_width'] is None:
        # crop the middle part according to the aspect ratio
        mid_row = int(math.floor(height * .5))
        desired_aspect_ratio = config['cnn_input_width'] / config['cnn_input_height']

        mid_point_x = math.floor(width / 2)
        mid_point_y = math.floor(height / 2)

        diameter = min(mid_point_x / desired_aspect_ratio, mid_point_y)

        left = round(mid_point_x - diameter * desired_aspect_ratio)
        right = round(mid_point_x + diameter * desired_aspect_ratio)
        top = round(mid_point_y - diameter)
        bottom = round(mid_point_y + diameter)
    else:
        # crop a fixed window
        mid_row = int(math.floor(height * .5))

        left = 0
        right = config['crop_width']
        top = mid_row - math.floor(config['crop_height'] / 2)
        bottom = mid_row + math.ceil(config['crop_height'] / 2)

    img = img[top:bottom, left:right]

    # TODO: DEBUG COPIED OVER
    pixel_max_value = (2**14) - 1

    # invert to dense light
    img = pixel_max_value - img

    # normalize
    img = img / pixel_max_value

    # resize
    img = cv2.resize(img, (config['cnn_input_width'], config['cnn_input_height']), interpolation=cv2.INTER_LINEAR)
    # TODO: DEBUG COPIED OVER END

    # # scale to [0, 1]
    # img = (img - img.min()) / (img.max() - img.min())

    # # invert to dense light
    # img = 1 - img

    # # remove the outliers
    # img_floor = np.quantile(img, 0.1) - 0.01
    # img_ceil = np.quantile(img, 0.9) + 0.1
    # img = np.clip(img, img_floor, img_ceil)

    # # scale to [0, 1]
    # img = (img - img.min()) / (img.max() - img.min())

    # gamma transformation
    # img = img**40

    # # resize
    # img = cv2.resize(img, (config['cnn_input_width'], config['cnn_input_height']), interpolation=cv2.INTER_LINEAR)

    # # find the breast mask: identify the largest bright island
    # mask = np.where(img > 0.5, np.uint8(1), np.uint8(0))
    # n_islands, island_map, island_stats, island_centroids = cv2.connectedComponentsWithStats(mask)
    # island_map_black_zero = (island_map + 1) * mask
    # count_res = zip(*np.unique(island_map_black_zero, return_counts=True))
    # count_res_no_black = [(x, c) for x, c in count_res if x > 0]
    # largest_island_index = max(count_res_no_black, key=lambda x: x[1])[0]
    # breast_mask = np.where(island_map_black_zero == largest_island_index, np.uint8(1), np.uint8(0))

    # # fill the outside uniformly with black
    # breast_mask_d = cv2.dilate(breast_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    # breast_mask_d_soften = cv2.GaussianBlur(breast_mask_d.astype(np.float32), (41, 41), 20)
    # img *= breast_mask_d_soften

    # # raise the floor
    # img = np.clip(img, 0.3, 1)

    # # scale to [0, 1]
    # img = (img - img.min()) / (img.max() - img.min())

    # # cv2.connectedComponentsWithStats(mask) = (
    # #    6,
    # #    array([[0, 0, 0, ..., 0, 0, 0],
    # #           [0, 0, 0, ..., 0, 0, 0],
    # #           [0, 0, 0, ..., 0, 0, 0],
    # #           ...,
    # #           [5, 5, 5, ..., 0, 0, 0],
    # #           [5, 5, 5, ..., 0, 0, 0],
    # #           [5, 5, 5, ..., 0, 0, 0]], dtype=int32),
    # #    array([[      0,       0,    2000,    2600, 3835743],
    # #           [      0,     370,     913,    1992, 1195523],
    # #           [   1310,    1972,     551,     589,  165462],
    # #           [   1686,    2014,      40,      34,     935],
    # #           [   1778,    2100,      34,      33,     902],
    # #           [      0,    2522,      39,      78,    1435]], dtype=int32),
    # #    array([[1174.81912709, 1222.93467107],
    # #           [ 354.61076449, 1405.65405768],
    # #           [1595.03977953, 2287.80779273],
    # #           [1707.57967914, 2030.70053476],
    # #           [1794.62527716, 2116.25166297],
    # #           [  11.5825784 , 2573.04738676]])
    # # )

    # # separate the high frequency signal from low
    # img_low = filters.gaussian_filter(img, max(round(config['cnn_input_width'] / 500), 1))
    # img_low = signal.medfilt2d(img_low, round(config['cnn_input_width'] / 60) * 2 + 1)
    # img_high = 200 * (img - img_low)
    # img_high = 1 / (1 + np.exp(-img_high))
    # # img_high = img_high / (1 + np.abs(img_high))
    # # img_high = (img_high + 1) / 2
    # img_high = (img_high - img_high.min()) / (img_high.max() - img_high.min())
    # img_low = img_low**3  # use gamma to make the low frequency channel darker
    # img = np.array([img_high, img_low])

    # # scale to [0, 1]
    # # img = (img - img.min()) / (img.max() - img.min())

    return img


def find_all_dicoms_for_ucsf():
    ucsf_crosswalk = pd.read_csv('raw_data/ucsf_crosswalk.csv', dtype={'linkage_id': str, 'acquisition_id': str})
    print(ucsf_crosswalk)
    dmeta = pd.DataFrame()
    dmeta['dicom_id'] = 'ucsf_' + ucsf_crosswalk['acquisition_id']
    dmeta['patient_id'] = 'ucsf_' + ucsf_crosswalk['linkage_id']
    dmeta['path_to_dicom'] = [
        f"{config['basedir_to_ucsf_dicoms']}/{id_}{config['filename_suffix']}.dcm" for id_ in ucsf_crosswalk['acquisition_id']
    ]

    pmeta = pd.DataFrame()
    pmeta['patient_id'] = dmeta['patient_id'].unique()

    return dmeta, pmeta
    # linkage_id  acquisition_id
    #         89        20796093
    #         89        20796094
    #         89        20796095
    #         89        20796096
    #        103          668264


def find_all_dicoms_for_mayo():
    dicoms = os.listdir(config['basedir_to_mayo_dicoms'])
    dicom_ids = [re.match(f"^(([A-F0-9]+)_[A-Z]+_[0-9]+)({config['filename_suffix']})?\\.dcm$", x) for x in dicoms]
    # 7D7F656B7E7C65657D79_LCC_20090618.dcm
    dicom_ids = [
        OrderedDict(
            [
                ('dicom_id', 'mayo_' + x[1]),
                ('patient_id', 'mayo_' + x[2]),
                ('path_to_dicom', f"{config['basedir_to_mayo_dicoms']}/{x[1]}{config['filename_suffix']}.dcm"),
            ]
        ) for x in dicom_ids if x is not None
    ]
    dmeta = pd.DataFrame.from_records(dicom_ids)

    pmeta = pd.DataFrame()
    pmeta['patient_id'] = dmeta['patient_id'].unique()

    return dmeta, pmeta


def process_one_dicom(task):
    dcm = pydicom.dcmread(task['path_to_dicom'])
    output_base_dir = task['config']['output_base_dir']
    row = task['row']
    if ('ViewModifierCodeSequence' in dcm.ViewCodeSequence[0] and len(dcm.ViewCodeSequence[0].ViewModifierCodeSequence) > 0):
        row['view_mod'] = True
    else:
        row['view_mod'] = False
    row['view_position'] = dcm.ViewPosition
    row['image_laterality'] = dcm.ImageLaterality
    row['pixel_intensity_relationship'] = dcm.PixelIntensityRelationship
    row['pixel_intensity_relationship_sign'] = dcm.PixelIntensityRelationshipSign
    row['window_center'] = dcm.WindowCenter if 'WindowCenter' in dcm else None
    row['window_width'] = dcm.WindowWidth if 'WindowWidth' in dcm else None
    row['image_laterality'] = dcm.ImageLaterality
    row['field_of_view_horizontal_flip'] = dcm.FieldOfViewHorizontalFlip
    img = dcm.pixel_array
    row['height'] = img.shape[0]
    row['width'] = img.shape[1]
    half_x = img.shape[1] // 2
    third_y = img.shape[0] // 3
    row['left_half_sum'] = img[third_y:-third_y, :half_x].sum()
    row['right_half_sum'] = img[third_y:-third_y, half_x:].sum()

    if config['dense_dark']:
        row['flipped'] = row['left_half_sum'] > row['right_half_sum']
    else:
        row['flipped'] = row['left_half_sum'] < row['right_half_sum']

    row['pixel_disagree_with_header'] = row['flipped'] != (
        (row['field_of_view_horizontal_flip'] == 'YES') == (row['image_laterality'] == 'L')
    )

    if row['flipped']:
        img = np.fliplr(img)

    row['acquisition_date'] = dcm.AcquisitionDate
    row['acquisition_time'] = dcm.AcquisitionTime

    # np.save(f"{output_base_dir}/npys/{row['dicom_id']}.npy", img)
    np.save(f"{output_base_dir}/npys_for_cnn/{row['dicom_id']}.npy", process_an_img_for_cnn(img))

    return {
        'id_': task['dicom_id'],
        # TODO: add mid-processing images into here
        'row': row,
    }


def merge_and_process(config):
    output_base_dir = config['output_base_dir']

    dmeta_mayo, pmeta_mayo = find_all_dicoms_for_mayo()
    dmeta_ucsf, pmeta_ucsf = find_all_dicoms_for_ucsf()
    print(f"len(dmeta_mayo) =\n{len(dmeta_mayo)}")
    print(f"len(dmeta_ucsf) =\n{len(dmeta_ucsf)}")
    print(f"len(pmeta_mayo) =\n{len(pmeta_mayo)}")
    print(f"len(pmeta_ucsf) =\n{len(pmeta_ucsf)}")

    dmeta = pd.concat([dmeta_mayo, dmeta_ucsf])
    print(f"len(dmeta) =\n{len(dmeta)}")

    pmeta = pd.concat([pmeta_mayo, pmeta_ucsf])
    print(f"len(pmeta) =\n{len(pmeta)}")

    # -----------------------------------------------------------------------------------------------------------
    # STEP 1: dmeta
    # -----------------------------------------------------------------------------------------------------------

    # dmeta = pd.concat(
    #     [
    #         process_ucsf_dmeta('raw_data/ucsf_meta_train.csv'),
    #         # process_ucsf_dmeta('raw_data/ucsf_meta_test.csv'),
    #         process_mayo_dmeta('raw_data/mayo_meta_train.csv'),
    #         # process_mayo_dmeta('raw_data/mayo_meta_test.csv'),
    #     ]
    # )
    dmeta = dmeta.set_index('dicom_id')

    # find out all rows whose DICOM files do not exist
    dmeta['dicom_exists'] = [os.path.isfile(p) for p in dmeta['path_to_dicom']]
    del_vec = ~dmeta['dicom_exists']
    if sum(del_vec) > 0:
        warn(f"The following {sum(del_vec)} DICOMs are excluded because their files are not found in the folder:")
        print(dmeta.loc[del_vec])
        dmeta.loc[del_vec].to_csv(f"{output_base_dir}/excluded/dicoms_file_not_found.csv")
    dmeta = dmeta.loc[~del_vec]

    # -----------------------------------------------------------------------------------------------------------
    # STEP 2: pmeta
    # -----------------------------------------------------------------------------------------------------------

    pmeta = pmeta.set_index('patient_id')

    pmeta_with_labels = pd.concat(
        [
            process_ucsf_pmeta('raw_data/ucsf_meta_train.csv').assign(source='ucsf', group='train'),
            process_ucsf_pmeta('raw_data/ucsf_meta_test.csv').assign(source='ucsf', group='test'),
            process_mayo_pmeta('raw_data/mayo_meta_train.csv').assign(source='mayo', group='train'),
            process_mayo_pmeta('raw_data/mayo_meta_test.csv').assign(source='mayo', group='test'),
        ]
    )[['patient_id', 'label', 'group']].set_index('patient_id')

    pmeta = pmeta.join(pmeta_with_labels)
    pmeta['label'] = pmeta['label'].astype("UInt8")
    pmeta['group'] = pmeta['group'].fillna('test2')

    pmeta['n_dicoms'] = dmeta.groupby('patient_id').size()
    # Delete patients who do not have at least 4 DICOMs
    del_vec = pmeta['n_dicoms'] < 4
    if sum(del_vec) > 0:
        warn(f"The following {sum(del_vec)} patients are excluded because they do not have at least 4 DICOMs:")
        print(pmeta.loc[del_vec])
        pmeta.loc[del_vec].to_csv(f"{output_base_dir}/excluded/patients_view_missing_1.csv")
    pmeta = pmeta.loc[~del_vec]
    # INFO: there are 564 patients with more than 4 dicoms

    if config['n_samples'] is not None:
        pmeta = pmeta.reset_index()
        pmeta = pmeta.sort_values(['n_dicoms', 'patient_id'])
        pmeta = pmeta.set_index('patient_id')
        pmeta = pmeta.sample(n=config['n_samples'])

    patient_ids = set(pmeta.index)
    del_vec = ~dmeta['patient_id'].isin(patient_ids)
    if sum(del_vec) > 0:
        warn(f"The following {sum(del_vec)} DICOMs are excluded because their patients have just been excluded:")
        print(dmeta.loc[del_vec])
        dmeta.loc[del_vec].to_csv(f"{output_base_dir}/excluded/dicoms_exclued_because_patients_view_missing_1.csv")
    dmeta = dmeta.loc[~del_vec]

    # -----------------------------------------------------------------------------------------------------------
    # STEP 3: Read each of the DICOM files, write down their key info from the
    #         headers, and preprocess them
    # -----------------------------------------------------------------------------------------------------------

    def process_dicoms_taskgen():
        for _, x in dmeta.reset_index().iterrows():
            yield {'row': dict(x), 'config': config}

    results = list(parallelize(process_one_dicom, process_dicoms_taskgen(), len_=len(dmeta)))
    dmeta = pd.DataFrame.from_records([x['row'] for x in results])
    debug(dmeta)

    del_vec = dmeta['view_mod']
    if sum(del_vec) > 0:
        warn(
            f"The following {sum(del_vec)} DICOMs are excluded because "
            f"they have a modified view (ViewModifierCodeSequence):"
        )
        print(dmeta.loc[del_vec])
        dmeta.loc[del_vec].to_csv(f"{output_base_dir}/excluded/dicoms_exclued_because_view_modified.csv")
    dmeta = dmeta.loc[~del_vec]
    dmeta = dmeta[[
        'dicom_id',
        'pixel_intensity_relationship',
        'pixel_intensity_relationship_sign',
        'window_center',
        'window_width',
        'patient_id',
        'image_laterality',
        'view_position',
        'flipped',
        'dicom_exists',
        'height',
        'width',
        'view_mod',
        'field_of_view_horizontal_flip',
        'left_half_sum',
        'right_half_sum',
        'acquisition_date',
        'acquisition_time',
        'path_to_dicom',
        'pixel_disagree_with_header',
    ]]
    dmeta = dmeta.sort_values([
        'patient_id',
        'image_laterality',
        'view_position',
    ])
    dmeta = dmeta.set_index('dicom_id')
    dmeta.to_csv(f"{output_base_dir}/dmeta.csv")

    # -----------------------------------------------------------------------------------------------------------
    # STEP 4: Go back to processing pmeta, now that we have much more info from
    #         the DICOM files.
    # -----------------------------------------------------------------------------------------------------------

    def select_dicom(df, image_laterality, view_position):
        df = df.loc[(df['image_laterality'] == image_laterality) & (df['view_position'] == view_position)]
        if len(df) == 0:
            return np.NaN
        elif len(df) > 1:
            debug("---")
            df = df.sort_values(['acquisition_date', 'acquisition_time'], ascending=False)
            debug(f"{df[['acquisition_date', 'acquisition_time']]}")
            return df.index[0]
        else:
            return df.index[0]

    for patient_id, dmeta_g in dmeta.groupby('patient_id'):
        pmeta.loc[patient_id, 'LCC'] = select_dicom(dmeta_g, 'L', 'CC')
        pmeta.loc[patient_id, 'RCC'] = select_dicom(dmeta_g, 'R', 'CC')
        pmeta.loc[patient_id, 'LMLO'] = select_dicom(dmeta_g, 'L', 'MLO')
        pmeta.loc[patient_id, 'RMLO'] = select_dicom(dmeta_g, 'R', 'MLO')

    del_vec = pmeta[['LCC', 'RCC', 'LMLO', 'RMLO']].isna().any(axis=1)
    if sum(del_vec) > 0:
        warn(
            f"The following {sum(del_vec)} patients are excluded because they "
            f"have at least one view missing (patients_view_missing.csv):"
        )
        print(pmeta.loc[del_vec])
        pmeta.loc[del_vec].to_csv(f"{output_base_dir}/excluded/patients_view_missing_2.csv")

    pmeta = pmeta.loc[~del_vec]
    pmeta = pmeta.reset_index()
    pmeta = pmeta.sort_values(['n_dicoms', 'patient_id'])
    pmeta = pmeta.set_index('patient_id')
    pmeta['n_dicoms'] = dmeta.groupby('patient_id').size()
    pmeta.to_csv(f"{output_base_dir}/pmeta.csv")

    print(output_base_dir)


def plot(config):
    base_dir = config['output_base_dir']

    pmeta = pd.read_csv(f"{base_dir}/pmeta.csv", index_col=0)
    dmeta = pd.read_csv(f"{base_dir}/dmeta.csv", index_col=0)

    # TODO: make sure that the four panels are sorted consistently
    # TODO: investigate the missing files at the end of the table

    output_dir = f"{base_dir}/contact_sheets_png"
    os.makedirs(output_dir, exist_ok=True)
    for i_chunk, pmeta_chunk in enumerate(chunk_df(pmeta, config['n_samples'])):
        img_list = []
        for patient_id, row in pmeta_chunk.iterrows():
            sub_img_list = []
            sub_img_df = dmeta.loc[dmeta['patient_id'] == patient_id]
            for dicom_id, row2 in sub_img_df.iterrows():
                img = np.load(f"{base_dir}/npys_for_cnn/{dicom_id}.npy")
                sub_img_list.append(img)
            sub_img_labels = ', '.join(f"{a}_{b}" for a, b in zip(sub_img_df['image_laterality'], sub_img_df['view_position']))
            label = '\n'.join([
                patient_id,
                *[f"{k} = {v}" for k, v in row.iteritems()],
                sub_img_labels,
            ])
            img_list.append((sub_img_list, label))
        display_imgs(
            img_list,
            save_as=f"{output_dir}/chunk_{i_chunk:0>2}.png",
            aspect_ratio=1.75,
            dpi=config['plot_dpi'],
            channel_order='first',
        )


def add_volpara_to_pmeta():
    pmeta = pd.read_csv('out_data/pres512/pmeta_for_proc2pres_remove_ucsf_6051.csv', index_col=0)
    volpara_df = pd.read_csv('out_data/volpara.csv')
    density_vec = volpara_df.set_index('dicom_id')['volumetric_breast_density']

    def avg(row):
        sum = 0
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            sum += density_vec[row[view]]
        return sum / 4

    pmeta['volpara_vd'] = pmeta.apply(avg, axis=1)
    pmeta.to_csv('out_data/pres512/pmeta_for_proc2pres_remove_ucsf_6051.csv')


def run():
    # find_all_dicoms_for_ucsf()
    # find_all_dicoms_for_mayo()
    merge_and_process(config)
    # plot(config)


if __name__ == '__main__':
    fire.Fire()
