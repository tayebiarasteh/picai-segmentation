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
                  resume=False, augment=False, experiment_name='name', modality=1, multimodal=True):
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
    if multimodal:
        model = UNet3D(n_in_channels=3, n_out_classes=1, firstdim=48)
    else:
        model = UNet3D(n_in_channels=1, n_out_classes=1, firstdim=48)
    # weight = torch.Tensor(params['class_weights'])
    weight = None

    loss_function = BinaryDiceLoss
    # loss_function = torch.nn.BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    train_dataset = data_loader_3D(cfg_path=cfg_path, mode='train', modality=modality, multimodal=multimodal)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    if valid:
        valid_dataset = data_loader_3D(cfg_path=cfg_path, mode='valid', modality=modality, multimodal=multimodal)
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



def main_evaluate_3D_bootstrap_pvalue(global_config_path="picai-segmentation/config/config.yaml", experiment_name1='central_exp_for_test',
                                      experiment_name2='central_exp_for_test', experiment1_epoch_num=100, experiment2_epoch_num=100, modality=1, multimodal1=True, multimodal2=True):
    """Evaluation for all the images using the labels and calculating metrics.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']


    # Changeable network parameters
    if multimodal1:
        model1 = UNet3D(n_in_channels=3, n_out_classes=1, firstdim=48)
    else:
        model1 = UNet3D(n_in_channels=1, n_out_classes=1, firstdim=48)

    # Generate test set
    test_dataset = data_loader_3D(cfg_path=cfg_path1, mode='test', modality=modality, multimodal=multimodal1)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['batch_size_testvlid'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1)
    predictor1.setup_model(model=model1, modelepoch=experiment1_epoch_num)
    test_F1_1, test_accuracy1, test_specificity1, test_sensitivity1, test_precision1 = predictor1.predict_only(test_loader)

    F1_list1, accuracy_list1, specificity_list1, sensitivity_list1, precision_list1 = predictor1.bootstrapper(test_F1_1, test_accuracy1, test_specificity1, test_sensitivity1, test_precision1, index_list)


    # Changeable network parameters
    if multimodal2:
        model2 = UNet3D(n_in_channels=3, n_out_classes=1, firstdim=48)
    else:
        model2 = UNet3D(n_in_channels=1, n_out_classes=1, firstdim=48)


    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']

    # Generate test set
    test_dataset = data_loader_3D(cfg_path=cfg_path2, mode='test', modality=modality, multimodal=multimodal2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params2['Network']['batch_size_testvlid'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    predictor2 = Prediction(cfg_path2)
    predictor2.setup_model(model=model2, modelepoch=experiment2_epoch_num)
    test_F1_2, test_accuracy2, test_specificity2, test_sensitivity2, test_precision2 = predictor2.predict_only(test_loader)

    F1_list2, accuracy_list2, specificity_list2, sensitivity_list2, precision_list2 = predictor2.bootstrapper(test_F1_2, test_accuracy2, test_specificity2, test_sensitivity2, test_precision2, index_list)


    ########################## P-Values ########################
    predictor2.pvalue_calculator(F1_list1, accuracy_list1, specificity_list1, sensitivity_list1, precision_list1, params1,
                      F1_list2, accuracy_list2, specificity_list2, sensitivity_list2, precision_list2, params2)
    ########################## P-Values ########################




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
    # main_train_3D(global_config_path=global_config_path, valid=True, resume=False, augment=True, experiment_name='name_noaugmen2')

    # main_evaluate_3D(global_config_path=global_config_path, experiment_name='name_noaugment', modelepoch=2)
    # main_predict_3D(global_config_path=global_config_path, experiment_name='name_noaugment', modelepoch=2)

    main_evaluate_3D_bootstrap_pvalue(global_config_path=global_config_path, experiment_name1='name_noaugment', experiment1_epoch_num=2,
                                      experiment_name2='name_noaugmen2', experiment2_epoch_num=2, multimodal1=True, multimodal2=True)
