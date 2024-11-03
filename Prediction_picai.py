"""
Created on Nov 2, 2024.
Prediction_picai.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os.path
import numpy as np
import torchmetrics
from tqdm import tqdm
import torch.nn.functional as F
import torchio as tio
import nibabel as nib

from config.serde import read_config

epsilon = 1e-15



class Prediction:
    def __init__(self, cfg_path):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.
        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def setup_model(self, model, model_file_name=None, modelepoch=1):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch2_" + model_file_name))
        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch" + str(modelepoch) + "_" + model_file_name))



    def evaluate_3D(self, test_loader):
        """Evaluation with metrics epoch
        Returns
        -------
        epoch_f1_score: float
            average test F1 score
        average_specificity: float
            average test specificity
        average_sensitivity: float
            average test sensitivity
        average_precision: float
            average test precision
        """
        self.model.eval()
        total_f1_score = []
        total_accuracy = []
        total_specifity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, (image, label) in enumerate(tqdm(test_loader)):
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

            ############ Evaluation metric calculation ########
            # Metrics calculation (macro) over the whole set
            label = label.int()

            confusioner = torchmetrics.ConfusionMatrix(num_classes=output_sigmoided.shape[1], multilabel=True).to(self.device)
            confusion = confusioner(output_sigmoided.squeeze(0,1).flatten(), label.squeeze(0).flatten())

            F1_disease = []
            accuracy_disease = []
            specifity_disease = []
            sensitivity_disease = []
            precision_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specifity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))
                precision_disease.append(TP / (TP + FP + epsilon))

            # Macro averaging
            total_f1_score.append(torch.stack(F1_disease))
            total_accuracy.append(torch.stack(accuracy_disease))
            total_specifity_score.append(torch.stack(specifity_disease))
            total_sensitivity_score.append(torch.stack(sensitivity_disease))
            total_precision_score.append(torch.stack(precision_disease))

        average_f1_score = torch.stack(total_f1_score).mean(0)
        average_accuracy = torch.stack(total_accuracy).mean(0)
        average_specifity = torch.stack(total_specifity_score).mean(0)
        average_sensitivity = torch.stack(total_sensitivity_score).mean(0)
        average_precision = torch.stack(total_precision_score).mean(0)

        return average_f1_score, average_accuracy, average_specifity, average_sensitivity, average_precision




    def predict_3D(self, image, label, label_metadata):
        """Prediction of one signle image

        Returns
        -------
        """
        self.model.eval()

        image = image.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            output_sigmoided = F.sigmoid(output)
            output_sigmoided = output_sigmoided[0, 0]
            output_sigmoided_classified = (output_sigmoided > 0.5).float()
            output_sigmoided_classified = output_sigmoided_classified.cpu().numpy()
            output_sigmoided_classified = output_sigmoided_classified.transpose(2,1,0)

            new_label = nib.Nifti1Image(output_sigmoided_classified, affine=label_metadata.affine, header=label_metadata.header)


        return new_label
