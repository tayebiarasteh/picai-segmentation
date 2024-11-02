"""
Created on Nov 2, 2024.
csv_data_preprocess_picai.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import random
from math import ceil
import glob
import SimpleITK as sitk

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')




class csv_preprocess_picai():
    def __init__(self, cfg_path="/picai-segmentation/config/config.yaml"):
        pass


    def initial_csv_creator(self):
        """Only include the ones that have cancer segmentation, both human and AI"""

        # Human
        file_path = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled/*.nii.gz'

        file_list = glob.glob(file_path)
        filtered_files = []

        for namefile in tqdm(file_list):
            image = nib.load(namefile)
            array = image.get_fdata()
            if array.sum() > 5:
                dim1, dim2, dim3 = array.shape
                filtered_files.append((os.path.basename(namefile), dim1, dim2, dim3))

        df = pd.DataFrame(filtered_files, columns=['filename', 'Dim1', 'Dim2', 'Dim3'])

        df = df.sort_values(by='filename')

        output_csv = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled_maserlist.csv'
        df.to_csv(output_csv, index=False)


        # AI
        file_path = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a/*.nii.gz'

        file_list = glob.glob(file_path)
        filtered_files = []

        for namefile in tqdm(file_list):
            image = nib.load(namefile)
            array = image.get_fdata()
            if array.sum() > 5:
                print(os.path.basename(namefile))
                dim1, dim2, dim3 = array.shape
                filtered_files.append((os.path.basename(namefile), dim1, dim2, dim3))

        df = pd.DataFrame(filtered_files, columns=['filename', 'Dim1', 'Dim2', 'Dim3'])

        df = df.sort_values(by='filename')

        output_csv = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/AI/AI_maserlist.csv'
        df.to_csv(output_csv, index=False)

        # combine them
        masterlist_df = pd.read_csv('/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/AI/AI_maserlist.csv')
        resampled_masterlist_df = pd.read_csv('/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled_maserlist.csv')
        resampled_filenames = set(resampled_masterlist_df['filename'])

        masterlist_df['label'] = masterlist_df['filename'].apply(lambda x: 'h' if x in resampled_filenames else 'a')

        updated_masterlist_path = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/human_expert/updated_maserlist.csv'
        masterlist_df.to_csv(updated_masterlist_path, index=False)


    def csv_selector_resampler_cropper(self, csv_path='/home/soroosh/Documents/datasets/PI-CAI/updated_masterlist.csv'):

        df = pd.read_csv(csv_path)
        base_path = '/home/soroosh/Documents/datasets/PI-CAI/picai_public'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            filename = filename.split('.nii.gz')[0]

            img_base_path = os.path.join(base_path, filename.split('_')[0], filename)
            t2w_path = img_base_path + "_t2w.mha"
            adc_path = img_base_path + "_adc.mha"
            hbv_path = img_base_path + "_hbv.mha"

            resampled_adc, resampled_hbv, original_t2w = self.resampler(adc_path, hbv_path, t2w_path)

            gland_segmentation_basepath = '/home/soroosh/Documents/datasets/PI-CAI/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23'
            gland_segmentation_path = os.path.join(gland_segmentation_basepath, row['filename'])

            if row['label'] == 'h':
                cancer_segmentation_basepath = '/home/soroosh/Documents/datasets/PI-CAI/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled'
            elif row['label'] == 'a':
                cancer_segmentation_basepath = '/home/soroosh/Documents/datasets/PI-CAI/picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a'
            cancer_segmentation_path = os.path.join(cancer_segmentation_basepath, row['filename'])

            self.cropper(gland_segmentation_path, cancer_segmentation_path, resampled_adc, resampled_hbv, original_t2w)




    def resampler(self, adc_path, hbv_path, t2w_path):
        t2w = sitk.ReadImage(t2w_path)
        adc = sitk.ReadImage(adc_path)
        hbv = sitk.ReadImage(hbv_path)

        adc_resampler = sitk.ResampleImageFilter()
        adc_resampler.SetReferenceImage(t2w)
        adc_resampler.SetInterpolator(sitk.sitkLinear)
        adc_resampler.SetOutputSpacing(t2w.GetSpacing())
        adc_resampler.SetSize(t2w.GetSize())
        adc_resampler.SetOutputDirection(t2w.GetDirection())
        adc_resampler.SetOutputOrigin(t2w.GetOrigin())
        resampled_adc = adc_resampler.Execute(adc)

        hbv_resampler = sitk.ResampleImageFilter()
        hbv_resampler.SetReferenceImage(t2w)
        hbv_resampler.SetInterpolator(sitk.sitkLinear)
        hbv_resampler.SetOutputSpacing(t2w.GetSpacing())
        hbv_resampler.SetSize(t2w.GetSize())
        hbv_resampler.SetOutputDirection(t2w.GetDirection())
        hbv_resampler.SetOutputOrigin(t2w.GetOrigin())
        resampled_hbv = hbv_resampler.Execute(hbv)

        return resampled_adc, resampled_hbv, t2w


    def cropper(self, gland_segmentation_path, cancer_segmentation_path, resampled_adc, resampled_hbv, original_t2w):

        patnum = os.path.basename(gland_segmentation_path).split('_')[0]

        output_base_path = '/home/soroosh/Documents/datasets/PI-CAI/Final'

        os.makedirs(os.path.join(output_base_path, str(patnum)), exist_ok=True)
        gland_segmentation_path_basename = os.path.basename(gland_segmentation_path)

        original_image_path = os.path.join(output_base_path, str(patnum), gland_segmentation_path_basename.replace('.nii.gz', '_t2w_original.mha'))
        cropped_t2w_path = os.path.join(output_base_path, str(patnum), gland_segmentation_path_basename.replace('.nii.gz', '_t2w_cropped.mha'))
        cropped_adc_path = os.path.join(output_base_path, str(patnum), gland_segmentation_path_basename.replace('.nii.gz', '_adc_cropped.mha'))
        cropped_hbv_path = os.path.join(output_base_path, str(patnum), gland_segmentation_path_basename.replace('.nii.gz', '_hbv_cropped.mha'))

        cropped_cancer_segmentation_path = os.path.join(output_base_path, str(patnum), os.path.basename(cancer_segmentation_path).replace('.nii.gz', '_cropped-label.nii.gz'))
        gland_segmentation = sitk.ReadImage(gland_segmentation_path)
        cancer_segmentation = sitk.ReadImage(cancer_segmentation_path)

        gland_segmentation_array = sitk.GetArrayFromImage(gland_segmentation)
        cancer_segmentation_array = sitk.GetArrayFromImage(cancer_segmentation)
        adc_array = sitk.GetArrayFromImage(resampled_adc)
        hbv_array = sitk.GetArrayFromImage(resampled_hbv)
        t2w_array = sitk.GetArrayFromImage(original_t2w)

        indices = np.where(gland_segmentation_array == 1)
        min_x, max_x = np.min(indices[2]), np.max(indices[2])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])

        # Crop
        cropped_cancer_segmentation_array = cancer_segmentation_array[:, min_y:max_y + 1, min_x:max_x + 1]
        cropped_adc_array = adc_array[:, min_y:max_y + 1, min_x:max_x + 1]
        cropped_hbv_array = hbv_array[:, min_y:max_y + 1, min_x:max_x + 1]
        cropped_t2w_array = t2w_array[:, min_y:max_y + 1, min_x:max_x + 1]

        cropped_cancer_segmentation = sitk.GetImageFromArray(cropped_cancer_segmentation_array)
        cropped_adc = sitk.GetImageFromArray(cropped_adc_array)
        cropped_hbv = sitk.GetImageFromArray(cropped_hbv_array)
        cropped_t2w = sitk.GetImageFromArray(cropped_t2w_array)

        cropped_cancer_segmentation.SetSpacing(cancer_segmentation.GetSpacing())
        cropped_cancer_segmentation.SetDirection(cancer_segmentation.GetDirection())

        cropped_adc.SetSpacing(original_t2w.GetSpacing())
        cropped_adc.SetDirection(original_t2w.GetDirection())
        cropped_hbv.SetSpacing(original_t2w.GetSpacing())
        cropped_hbv.SetDirection(original_t2w.GetDirection())
        cropped_t2w.SetSpacing(original_t2w.GetSpacing())
        cropped_t2w.SetDirection(original_t2w.GetDirection())

        origin = original_t2w.TransformContinuousIndexToPhysicalPoint([float(min_x), float(min_y), 0.0])
        cropped_adc.SetOrigin(origin)
        cropped_hbv.SetOrigin(origin)
        cropped_t2w.SetOrigin(origin)
        cropped_cancer_segmentation.SetOrigin(origin)

        sitk.WriteImage(cropped_cancer_segmentation, cropped_cancer_segmentation_path)
        sitk.WriteImage(cropped_t2w, cropped_t2w_path)
        sitk.WriteImage(cropped_adc, cropped_adc_path)
        sitk.WriteImage(cropped_hbv, cropped_hbv_path)
        sitk.WriteImage(original_t2w, original_image_path)






if __name__ == '__main__':
    handler = csv_preprocess_picai()
    handler.csv_selector_resampler_cropper()
