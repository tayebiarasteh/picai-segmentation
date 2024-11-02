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
        self.params = read_config(cfg_path)


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






class Crop():
    def __init__(self, cfg_path="/picai-segmentation/config/config.yaml"):
        """
        Cropping the all the images and segmentations around the brain
        Parameters
        ----------
        cfg_path
        """
        pass


    def cropper(self):
        gland_segmentation_path = '/home/soroosh/Downloads/picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b/10005_1000005.nii.gz'
        cancer_segmentation_path = '/home/soroosh/Downloads/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled/10005_1000005.nii.gz'
        image_path = '/home/soroosh/Downloads/picai_public_images_fold4/10005/10005_1000005_t2w.mha'

        patnum = os.path.basename(image_path).split('_')[0]

        output_base_path = '/home/soroosh/Documents/datasets/PI-CAI'

        os.makedirs(os.path.join(output_base_path, str(patnum)), exist_ok=True)

        original_image_path = os.path.join(output_base_path, str(patnum), os.path.basename(image_path).replace('t2w.mha', 't2w_original.mha'))
        cropped_image_path = os.path.join(output_base_path, str(patnum), os.path.basename(image_path).replace('t2w.mha', 't2w_cropped.mha'))
        cropped_cancer_segmentation_path = os.path.join(output_base_path, str(patnum), os.path.basename(cancer_segmentation_path).replace('.nii.gz', '_cropped-label.nii.gz'))

        gland_segmentation = sitk.ReadImage(gland_segmentation_path)
        cancer_segmentation = sitk.ReadImage(cancer_segmentation_path)
        img = sitk.ReadImage(image_path)

        gland_segmentation_array = sitk.GetArrayFromImage(gland_segmentation)
        cancer_segmentation_array = sitk.GetArrayFromImage(cancer_segmentation)
        img_array = sitk.GetArrayFromImage(img)

        indices = np.where(gland_segmentation_array == 1)
        min_x, max_x = np.min(indices[2]), np.max(indices[2])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])

        # Crop
        cropped_cancer_segmentation_array = cancer_segmentation_array[:, min_y:max_y + 1, min_x:max_x + 1]
        cropped_img_array = img_array[:, min_y:max_y + 1, min_x:max_x + 1]

        cropped_cancer_segmentation = sitk.GetImageFromArray(cropped_cancer_segmentation_array)
        cropped_img = sitk.GetImageFromArray(cropped_img_array)

        cropped_cancer_segmentation.SetSpacing(cancer_segmentation.GetSpacing())
        cropped_cancer_segmentation.SetDirection(cancer_segmentation.GetDirection())

        cropped_img.SetSpacing(img.GetSpacing())
        cropped_img.SetDirection(img.GetDirection())

        origin = img.TransformContinuousIndexToPhysicalPoint([float(min_x), float(min_y), 0.0])
        cropped_img.SetOrigin(origin)
        cropped_cancer_segmentation.SetOrigin(origin)

        sitk.WriteImage(cropped_cancer_segmentation, cropped_cancer_segmentation_path)
        sitk.WriteImage(cropped_img, cropped_image_path)
        sitk.WriteImage(img, original_image_path)






if __name__ == '__main__':
    # handler = csv_preprocess_picai()
    cropclass = Crop()
    cropclass.cropper()
