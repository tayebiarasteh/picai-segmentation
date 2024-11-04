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
import nibabel as nib
import pandas as pd

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



    def predict_only(self, test_loader):
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

        average_f1_score = torch.stack(total_f1_score)
        average_accuracy = torch.stack(total_accuracy)
        average_specifity = torch.stack(total_specifity_score)
        average_sensitivity = torch.stack(total_sensitivity_score)
        average_precision = torch.stack(total_precision_score)

        return average_f1_score.cpu().numpy(), average_accuracy.cpu().numpy(), average_specifity.cpu().numpy(), average_sensitivity.cpu().numpy(), average_precision.cpu().numpy()



    def bootstrapper(self, test_F1, test_accuracy, test_specificity, test_sensitivity, test_precision, index_list):
        self.model.eval()
        F1_list = []
        accuracy_list = []
        specificity_list = []
        sensitivity_list = []
        precision_list = []

        test_F1 = test_F1.flatten()
        test_accuracy = test_accuracy.flatten()
        test_specificity = test_specificity.flatten()
        test_sensitivity = test_sensitivity.flatten()
        test_precision = test_precision.flatten()

        print('bootstrapping ... \n')

        for counter in range(1000):

            new_F1 = test_F1[index_list[counter]]
            new_accuracy = test_accuracy[index_list[counter]]
            new_specificity = test_specificity[index_list[counter]]
            new_sensitivity = test_sensitivity[index_list[counter]]
            new_precision = test_precision[index_list[counter]]

            F1_list.append(new_F1.mean())
            accuracy_list.append(new_accuracy.mean())
            specificity_list.append(new_specificity.mean())
            sensitivity_list.append(new_sensitivity.mean())
            precision_list.append(new_precision.mean())

        F1_list = np.stack(F1_list)
        accuracy_list = np.stack(accuracy_list)
        specificity_list = np.stack(specificity_list)
        sensitivity_list = np.stack(sensitivity_list)
        precision_list = np.stack(precision_list)


        print('------------------------------------------------------'
              '----------------------------------')
        print('\t experiment:' + self.params['experiment_name'] + '\n')

        print(f'\t Dice: {F1_list.mean() * 100:.2f}% ± {F1_list.std() * 100:.2f} [{np.percentile(F1_list, 2.5) * 100:.2f}%, {np.percentile(F1_list, 97.5) * 100:.2f}%] | Accuracy: {accuracy_list.mean() * 100:.2f}% ± {accuracy_list.std() * 100:.2f} [{np.percentile(accuracy_list, 2.5) * 100:.2f}%, {np.percentile(accuracy_list, 97.5) * 100:.2f}%] '
              f' | Specificity: {specificity_list.mean() * 100:.2f}% ± {specificity_list.std() * 100:.2f} [{np.percentile(specificity_list, 2.5) * 100:.2f}%, {np.percentile(specificity_list, 97.5) * 100:.2f}%] '
              f' | Sensitivity (Recall): {sensitivity_list.mean() * 100:.2f}% ± {sensitivity_list.std() * 100:.2f} [{np.percentile(sensitivity_list, 2.5) * 100:.2f}%, {np.percentile(sensitivity_list, 97.5) * 100:.2f}%]  | Precision: {precision_list.mean() * 100:.2f}% ± {precision_list.std() * 100:.2f} [{np.percentile(precision_list, 2.5) * 100:.2f}%, {np.percentile(precision_list, 97.5) * 100:.2f}%] \n')

        print('------------------------------------------------------'
              '----------------------------------')

        # saving the stats
        msg = f'\n\n----------------------------------------------------------------------------------------\n' \
              '\t experiment:' + self.params['experiment_name'] + '\n\n' \
              f'\t Dice: {F1_list.mean() * 100:.2f}% ± {F1_list.std() * 100:.2f} [{np.percentile(F1_list, 2.5) * 100:.2f}%, {np.percentile(F1_list, 97.5) * 100:.2f}%] | Accuracy: {accuracy_list.mean() * 100:.2f}% ± {accuracy_list.std() * 100:.2f} [{np.percentile(accuracy_list, 2.5) * 100:.2f}%, {np.percentile(accuracy_list, 97.5) * 100:.2f}%] ' \
              f' | Specificity: {specificity_list.mean() * 100:.2f}% ± {specificity_list.std() * 100:.2f} [{np.percentile(specificity_list, 2.5) * 100:.2f}%, {np.percentile(specificity_list, 97.5) * 100:.2f}%] ' \
              f' | Sensitivity (Recall): {sensitivity_list.mean() * 100:.2f}% ± {sensitivity_list.std() * 100:.2f} [{np.percentile(sensitivity_list, 2.5) * 100:.2f}%, {np.percentile(sensitivity_list, 97.5) * 100:.2f}%]  | Precision: {precision_list.mean() * 100:.2f}% ± {precision_list.std() * 100:.2f} [{np.percentile(precision_list, 2.5) * 100:.2f}%, {np.percentile(precision_list, 97.5) * 100:.2f}%] \n\n'

        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

        df = pd.DataFrame({'Dice_mean': F1_list *100, 'Accuracy_mean': accuracy_list *100, 'Specificity_mean': specificity_list *100, 'Sensitivity_mean': sensitivity_list *100, 'Precision_mean': precision_list *100})

        df.to_csv(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/bootstrapped_results.csv', sep=',', index=False)

        return F1_list, accuracy_list, specificity_list, sensitivity_list, precision_list



    def pvalue_calculator(self, F1_list1, accuracy_list1, specificity_list1, sensitivity_list1, precision_list1, params1,
                          F1_list2, accuracy_list2, specificity_list2, sensitivity_list2, precision_list2, params2):

        ########################## Dice ########################
        counter = F1_list1 > F1_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\n\nDice p-value: {ratio1}; model 1 significantly higher Dice than model 2')
        else:
            counter = F1_list2 > F1_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\n\nDice p-value: {ratio2}; model 2 significantly higher Dice than model 1')
            else:
                print(f'\n\nDice p-value: {ratio1}; models NOT significantly different in terms of Dice')

        counter = F1_list1 > F1_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\n\nDice p-value: {ratio1}; model 1 significantly higher Dice than model 2'
        else:
            counter = F1_list2 > F1_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\n\nDice p-value: {ratio2}; model 2 significantly higher Dice than model 1'
            else:
                msg = f'\n\nDice p-value: {ratio1}; models NOT significantly different in terms of Dice'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        ########################## Dice ########################


        ########################## Accuracy ########################
        counter = accuracy_list1 > accuracy_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\n\nAccuracy p-value: {ratio1}; model 1 significantly higher Accuracy than model 2')
        else:
            counter = accuracy_list2 > accuracy_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\n\nAccuracy p-value: {ratio2}; model 2 significantly higher Accuracy than model 1')
            else:
                print(f'\n\nAccuracy p-value: {ratio1}; models NOT significantly different in terms of Accuracy')

        counter = accuracy_list1 > accuracy_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\n\nAccuracy p-value: {ratio1}; model 1 significantly higher Accuracy than model 2'
        else:
            counter = accuracy_list2 > accuracy_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\n\nAccuracy p-value: {ratio2}; model 2 significantly higher Accuracy than model 1'
            else:
                msg = f'\n\nAccuracy p-value: {ratio1}; models NOT significantly different in terms of Accuracy'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        ########################## Accuracy ########################


        ########################## Specificity ########################
        counter = specificity_list1 > specificity_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\n\nSpecificity p-value: {ratio1}; model 1 significantly higher Specificity than model 2')
        else:
            counter = specificity_list2 > specificity_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\n\nSpecificity p-value: {ratio2}; model 2 significantly higher Specificity than model 1')
            else:
                print(f'\n\nSpecificity p-value: {ratio1}; models NOT significantly different in terms of Specificity')

        counter = specificity_list1 > specificity_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\n\nSpecificity p-value: {ratio1}; model 1 significantly higher Specificity than model 2'
        else:
            counter = specificity_list2 > specificity_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\n\nSpecificity p-value: {ratio2}; model 2 significantly higher Specificity than model 1'
            else:
                msg = f'\n\nSpecificity p-value: {ratio1}; models NOT significantly different in terms of Specificity'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        ########################## Specificity ########################


        ########################## Sensitivity ########################
        counter = sensitivity_list1 > sensitivity_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\n\nSensitivity p-value: {ratio1}; model 1 significantly higher Sensitivity than model 2')
        else:
            counter = sensitivity_list2 > sensitivity_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\n\nSensitivity p-value: {ratio2}; model 2 significantly higher Sensitivity than model 1')
            else:
                print(f'\n\nSensitivity p-value: {ratio1}; models NOT significantly different in terms of Sensitivity')

        counter = sensitivity_list1 > sensitivity_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\n\nSensitivity p-value: {ratio1}; model 1 significantly higher Sensitivity than model 2'
        else:
            counter = sensitivity_list2 > sensitivity_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\n\nSensitivity p-value: {ratio2}; model 2 significantly higher Sensitivity than model 1'
            else:
                msg = f'\n\nSensitivity p-value: {ratio1}; models NOT significantly different in terms of Sensitivity'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        ########################## Sensitivity ########################


        ########################## Precision ########################
        counter = precision_list1 > precision_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\n\nPrecision p-value: {ratio1}; model 1 significantly higher Precision than model 2')
        else:
            counter = precision_list2 > precision_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\n\nPrecision p-value: {ratio2}; model 2 significantly higher Precision than model 1')
            else:
                print(f'\n\nPrecision p-value: {ratio1}; models NOT significantly different in terms of Precision')

        counter = precision_list1 > precision_list2
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\n\nPrecision p-value: {ratio1}; model 1 significantly higher Precision than model 2'
        else:
            counter = precision_list2 > precision_list1
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\n\nPrecision p-value: {ratio2}; model 2 significantly higher Precision than model 1'
            else:
                msg = f'\n\nPrecision p-value: {ratio1}; models NOT significantly different in terms of Precision'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        ########################## Sensitivity ########################


