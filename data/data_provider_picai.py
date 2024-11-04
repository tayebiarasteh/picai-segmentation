"""
Created on Nov 2, 2024.
data_provider_picai.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from data.augmentation_picai import patch_cropper

from config.serde import read_config






class data_loader_3D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', modality=1, multimodal=True):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train

        modality: int
            modality of the MR sequence
            1: t2w
            2: adc
            3: hbv
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.modality = int(modality)
        self.multimodal = multimodal
        self.mode = mode

        # org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_masterlist.csv"), sep=',')
        # org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_masterlist_novalid.csv"), sep=',')
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_masterlist_short.csv"), sep=',')

        if mode=='train':
            self.subset_df = org_df[org_df['subset'] == 'train']
        elif mode == 'valid':
            self.subset_df = org_df[org_df['subset'] == 'valid']
        elif mode == 'test':
            self.subset_df = org_df[org_df['subset'] == 'test']

        self.file_path_list = list(self.subset_df['filename'])


        # filename = self.file_path_list[0]


    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        filename = self.file_path_list[idx]
        filename = filename.split('.nii.gz')[0]

        img_base_path = os.path.join(self.file_base_dir, 'Final', filename.split('_')[0], filename)
        label_path = img_base_path + "_cropped-label.nii.gz"

        label = nib.load(label_path)
        label_array = label.get_fdata() # (w, h, d)
        label_array = label_array.transpose(2, 1, 0) # (d, h, w)
        label_array = label_array.astype(np.int) # (d, h, w)
        label_array = np.where(label_array > 1, 1, label_array)

        if self.multimodal:
            t2w_path = img_base_path + "_t2w_cropped.mha"
            adc_path = img_base_path + "_adc_cropped.mha"
            hbv_path = img_base_path + "_hbv_cropped.mha"

            t2w = sitk.ReadImage(t2w_path)
            t2w_array = sitk.GetArrayFromImage(t2w)  # (d, h, w)

            adc = sitk.ReadImage(adc_path)
            adc_array = sitk.GetArrayFromImage(adc)  # (d, h, w)

            hbv = sitk.ReadImage(hbv_path)
            hbv_array = sitk.GetArrayFromImage(hbv)  # (d, h, w)

            # normalization
            normalized_t2w = self.irm_min_max_preprocess(t2w_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_t2w = normalized_t2w.transpose(2, 0, 1)  # (d, h, w)

            normalized_adc = self.irm_min_max_preprocess(adc_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_adc = normalized_adc.transpose(2, 0, 1)  # (d, h, w)

            normalized_hbv = self.irm_min_max_preprocess(hbv_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_hbv = normalized_hbv.transpose(2, 0, 1)  # (d, h, w)

            normalized_img_resized = np.stack((normalized_t2w, normalized_adc, normalized_hbv))  # (c=3, d, h, w)
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (c=3, d, h, w)

        else:
            if self.modality == 1:
                path_file = img_base_path + "_t2w_cropped.mha"
            elif self.modality == 2:
                path_file = img_base_path + "_adc_cropped.mha"
            elif self.modality == 3:
                path_file = img_base_path + "_hbv_cropped.mha"

            img = sitk.ReadImage(path_file)
            img_array = sitk.GetArrayFromImage(img)  # (d, h, w)
            # normalization
            normalized_img = self.irm_min_max_preprocess(img_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_img = normalized_img.transpose(2, 0, 1)  # (d, h, w)

            normalized_img_resized = normalized_img
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
            normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        label = torch.from_numpy(label_array)  # (d, h, w)

        # normalized_img_resized = normalized_img_resized.half() # float16
        normalized_img_resized = normalized_img_resized.float() # float32
        # label = label.int() # int32
        label = label.float() # float32 for cross entropy loss

        if self.mode=='train':
            normalized_img_resized, label = patch_cropper(normalized_img_resized.unsqueeze(0), label.unsqueeze(0), self.cfg_path)
            normalized_img_resized = normalized_img_resized.squeeze(0)
            label = label.squeeze(0)

        return normalized_img_resized, label



    def data_normalization_mean_std(self, image):
        """subtarcting mean and std for each individual patient and modality
        mean and std only over the tumor region

        Parameters
        ----------
        image: numpy array
            The raw input image
        Returns
        -------
        normalized_img: numpy array
            The normalized image
        """
        mean = image[image > 0].mean()
        std = image[image > 0].std()

        if self.outzero_normalization:
            image[image < 0] = -1000

        normalized_img = (image - mean) / std

        if self.outzero_normalization:
            normalized_img[normalized_img < -100] = 0

        return normalized_img



    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)

        min_ = np.min(image)
        max_ = np.max(image)
        scale = max_ - min_
        image = (image - min_) / scale

        return image



class data_loader_without_label_3D():
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, modality=1, multimodal=True):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.modality = int(modality)
        self.multimodal = multimodal



    def provide_test_without_label(self, file_path):
        """
        Parameters
        ----------
        """
        filename = file_path.split('.nii.gz')[0]

        img_base_path = os.path.join(self.file_base_dir, 'Final', filename.split('_')[0], filename)
        label_path = img_base_path + "_cropped-label.nii.gz"

        label_metadata = nib.load(label_path)
        label_array = label_metadata.get_fdata() # (w, h, d)
        label_array = label_array.transpose(2, 1, 0) # (d, h, w)
        label_array = label_array.astype(np.int) # (d, h, w)
        label_array = np.where(label_array > 1, 1, label_array)

        if self.multimodal:
            t2w_path = img_base_path + "_t2w_cropped.mha"
            adc_path = img_base_path + "_adc_cropped.mha"
            hbv_path = img_base_path + "_hbv_cropped.mha"

            t2w = sitk.ReadImage(t2w_path)
            t2w_array = sitk.GetArrayFromImage(t2w)  # (d, h, w)

            adc = sitk.ReadImage(adc_path)
            adc_array = sitk.GetArrayFromImage(adc)  # (d, h, w)

            hbv = sitk.ReadImage(hbv_path)
            hbv_array = sitk.GetArrayFromImage(hbv)  # (d, h, w)

            # normalization
            normalized_t2w = self.irm_min_max_preprocess(t2w_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_t2w = normalized_t2w.transpose(2, 0, 1)  # (d, h, w)

            normalized_adc = self.irm_min_max_preprocess(adc_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_adc = normalized_adc.transpose(2, 0, 1)  # (d, h, w)

            normalized_hbv = self.irm_min_max_preprocess(hbv_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_hbv = normalized_hbv.transpose(2, 0, 1)  # (d, h, w)

            normalized_img_resized = np.stack((normalized_t2w, normalized_adc, normalized_hbv))  # (c=3, d, h, w)
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (c=3, d, h, w)

        else:
            if self.modality == 1:
                path_file = img_base_path + "_t2w_cropped.mha"
            elif self.modality == 2:
                path_file = img_base_path + "_adc_cropped.mha"
            elif self.modality == 3:
                path_file = img_base_path + "_hbv_cropped.mha"

            img = sitk.ReadImage(path_file)
            img_array = sitk.GetArrayFromImage(img)  # (d, h, w)
            # normalization
            normalized_img = self.irm_min_max_preprocess(img_array.transpose(1, 2, 0))  # (h, w, d)
            normalized_img = normalized_img.transpose(2, 0, 1)  # (d, h, w)

            normalized_img_resized = normalized_img
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
            normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        label = torch.from_numpy(label_array)  # (d, h, w)

        # normalized_img_resized = normalized_img_resized.half() # float16
        normalized_img_resized = normalized_img_resized.float() # float32
        # label = label.int() # int32
        label = label.float() # float32 for cross entropy loss


        return normalized_img_resized, label, label_metadata



    def data_normalization_mean_std(self, image):
        """subtarcting mean and std for each individual patient and modality
        mean and std only over the tumor region

        Parameters
        ----------
        image: numpy array
            The raw input image
        Returns
        -------
        normalized_img: numpy array
            The normalized image
        """
        mean = image[image > 0].mean()
        std = image[image > 0].std()

        if self.outzero_normalization:
            image[image < 0] = -1000

        normalized_img = (image - mean) / std

        if self.outzero_normalization:
            normalized_img[normalized_img < -100] = 0

        return normalized_img



    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)

        min_ = np.min(image)
        max_ = np.max(image)
        scale = max_ - min_
        image = (image - min_) / scale

        return image

