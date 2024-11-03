"""
Created on Nov 2, 2024.
main_3D_picai.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import nibabel as nib
from math import floor
from sklearn import metrics
import pandas as pd

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_picai import Training
from Prediction_picai import Prediction
from data.data_provider_picai import data_loader_3D, data_loader_without_label_3D
from models.UNet3D import UNet3D
from models.Diceloss import BinaryDiceLoss

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15




def main_train_3D(global_config_path="picai-segmentation/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name'):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="federated_he/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = UNet3D(n_out_classes=1, firstdim=48)
    # weight = torch.Tensor(params['class_weights'])
    weight = None

    # loss_function = BinaryDiceLoss
    loss_function = torch.nn.BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    train_dataset = data_loader_3D(cfg_path=cfg_path, mode='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    if valid:
        valid_dataset = data_loader_3D(cfg_path=cfg_path, mode='valid')
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size_testvlid'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=6)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, augment=augment)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)




def main_evaluate_3D(global_config_path="picai-segmentation/config/config.yaml", experiment_name='name', modelepoch=2):
    """Evaluation for all the images using the labels and calculating metrics.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=1, firstdim=48)

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model, modelepoch=modelepoch)

    # Generate test set
    test_dataset = data_loader_3D(cfg_path=cfg_path, mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size_testvlid'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    test_F1, test_accuracy, test_specificity, test_sensitivity, test_precision = predictor.evaluate_3D(test_loader)

    ### evaluation metrics
    print(f'\t Dice: {test_F1.item() * 100:.2f}% | Accuracy: {test_accuracy.item() * 100:.2f}%'
        f' | Specificity: {test_specificity.item() * 100:.2f}%'
        f' | Sensitivity (Recall): {test_sensitivity.item() * 100:.2f}% | Precision: {test_precision.item() * 100:.2f}%\n')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the training and validation stats
    msg = f'\n\n----------------------------------------------------------------------------------------\n' \
          f'Dice: {test_F1.item() * 100:.2f}% | Accuracy: {test_accuracy.item() * 100:.2f}% ' \
          f' | Specificity: {test_specificity.item() * 100:.2f}%' \
          f' | Sensitivity (Recall): {test_sensitivity.item() * 100:.2f}% | Precision: {test_precision.item() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results', 'a') as f:
        f.write(msg)



def main_predict_3D(global_config_path="picai-segmentation/config/config.yaml", experiment_name='name', modelepoch=2):
    """Evaluation for all the images using the labels and calculating metrics.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=1, firstdim=48)

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model, modelepoch=modelepoch)

    # Generate test set
    test_dataset = data_loader_without_label_3D(cfg_path=cfg_path)


    file_base_dir = params['file_path']
    org_df = pd.read_csv(os.path.join(file_base_dir, "final_masterlist.csv"), sep=',')
    subset_df = org_df[org_df['subset'] == 'test']
    file_path_list = list(subset_df['filename'])
    outputdir = os.path.join(params['target_dir'], params['output_data_path'])

    for file_path in tqdm(file_path_list):
        image, label, label_metadata = test_dataset.provide_test_without_label(file_path)
        new_label = predictor.predict_3D(image, label, label_metadata)
        output_path = os.path.join(outputdir, file_path)
        output_path = output_path.replace('.nii.gz', '_predicted-label.nii.gz')
        nib.save(new_label, output_path)






if __name__ == '__main__':
    global_config_path = "/home/soroosh/Documents/Repositories/picai-segmentation/config/config.yaml"
    # delete_experiment(global_config_path=global_config_path, experiment_name='name_noaugment')
    # main_train_3D(global_config_path=global_config_path, valid=True, resume=False, augment=True, experiment_name='name_noaugment')

    # main_evaluate_3D(global_config_path=global_config_path, experiment_name='name_noaugment', modelepoch=2)
    main_predict_3D(global_config_path=global_config_path, experiment_name='name_noaugment', modelepoch=2)