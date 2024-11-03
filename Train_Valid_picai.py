"""
Created on Nov 2, 2024.
Train_Valid_picai.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import torchmetrics
import torch.nn.functional as F

from config.serde import read_config, write_config
from data.augmentation_picai import random_augment

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training:
    def __init__(self, cfg_path, num_epochs=10, resume=False, augment=False):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_epochs: int
            Total number of epochs for training

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_epochs = num_epochs
        self.augment = augment

        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.best_loss = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))


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


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        # self.model = self.model.half() # float16

        if not weight==None:
            # self.loss_weight = weight.to(self.device)
            # self.loss_function = loss_function(self.loss_weight) # for binary
            # self.loss_function = loss_function(pos_weight=self.loss_weight) # for multi label
            self.loss_function = loss_function()
        else:
            self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model'] = model.__name__
        # self.model_info['optimiser'] = optimiser.__name__
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['num_epochs'] = self.num_epochs
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, loss_function, weight=None):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        # self.loss_weight = weight.to(self.device)
        # self.loss_function = loss_function(self.loss_weight)
        self.loss_function = loss_function()
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)



    def train_epoch(self, train_loader, valid_loader=None):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            start_time = time.time()

            for idx, (image, label) in enumerate(train_loader):
                self.model.train()

                # if we would like to have data augmentation during training
                if self.augment:
                    image, label = random_augment(image, label, self.cfg_path)

                image = image.to(self.device)
                label = label.to(self.device)

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):

                    output = self.model(image)

                    loss = self.loss_function(F.sigmoid(output.squeeze(1)), label) # for binary loss (both CE and Dice)

                    loss.backward()
                    self.optimiser.step()

                    batch_loss += loss.item()

            # Prints train loss after number of steps specified.
            train_loss = batch_loss / len(train_loader)
            self.writer.add_scalar('Train_loss_avg', train_loss, self.epoch)

            # Saves information about training to config file
            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ######## Save a checkpoint every epoch ########
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(),
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))
            ######## Save a checkpoint every epoch ########

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_F1, valid_accuracy, valid_specificity, valid_sensitivity, valid_precision = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_accuracy=valid_accuracy, valid_specificity=valid_specificity,
                                            valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, valid_loss=valid_loss,
                                        valid_F1=valid_F1, valid_accuracy=valid_accuracy, valid_specificity= valid_specificity,
                                        valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                else:
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss)



    def valid_epoch(self, valid_loader):
        """Validation epoch

        """
        self.model.eval()
        total_loss = 0.0
        total_f1_score = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, (image, label) in enumerate(valid_loader):

            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = self.loss_function(F.sigmoid(output.squeeze(1)), label)  # for binary loss (both CE and Dice)

                # output_sigmoided = F.sigmoid(output.permute(0, 2, 3, 4, 1))
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

            ############ Evaluation metric calculation ########
            total_loss += loss.item()

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
            total_specificity_score.append(torch.stack(specifity_disease))
            total_sensitivity_score.append(torch.stack(sensitivity_disease))
            total_precision_score.append(torch.stack(precision_disease))

        average_loss = total_loss / len(valid_loader)
        average_f1_score = torch.stack(total_f1_score).mean(0)
        average_accuracy = torch.stack(total_accuracy).mean(0)
        average_specifity = torch.stack(total_specificity_score).mean(0)
        average_sensitivity = torch.stack(total_sensitivity_score).mean(0)
        average_precision = torch.stack(total_precision_score).mean(0)

        return average_loss, average_f1_score, average_accuracy, average_specifity, average_sensitivity, average_precision



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, valid_loss=None, valid_F1=None, valid_accuracy=None,
                       valid_specificity=None, valid_sensitivity=None, valid_precision=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_accuracy: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specificity: float
            validation specificity of the model

        valid_loss: float
            validation loss of the model
        """
        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best loss
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'model_info': self.model_info, 'best_loss': self.best_loss},
                   os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        # Saving every couple of epochs
        if (self.epoch) % self.params['display_stats_freq'] == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                    'epoch{}_'.format(self.epoch) + self.params['trained_model_name']))


        print('------------------------------------------------------'
              '----------------------------------')
        print(f'epoch: {self.epoch} | '
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs:.1f}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | Dice: {valid_F1.item() * 100:.2f}% | Accuracy: {valid_accuracy.item() * 100:.2f}%'
            f' | Specificity: {valid_specificity.item() * 100:.2f}%'
            f' | Sensitivity (Recall): {valid_sensitivity.item() * 100:.2f}% | Precision: {valid_precision.item() * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'\n\n----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.1f}s | ' \
                  f'\n\n\tTrain loss: {train_loss:.4f}' \
                   f'Val. loss: {valid_loss:.4f} | Dice: {valid_F1.item() * 100:.2f}% | Accuracy: {valid_accuracy.item() * 100:.2f}% ' \
                   f' | Specificity: {valid_specificity.item() * 100:.2f}%' \
                   f' | Sensitivity (Recall): {valid_sensitivity.item() * 100:.2f}% | Precision: {valid_precision.item() * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.1f}s\n\n\ttrain loss: {train_loss:.4f}\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_loss=None, valid_F1=None, valid_accuracy=None, valid_specificity=None, valid_sensitivity=None, valid_precision=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        valid_accuracy: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specificity: float
            validation specificity of the model

        valid_loss: float
            validation loss of the model
        """
        if valid_loss is not None:
            self.writer.add_scalar('Valid_loss', valid_loss, self.epoch)
            self.writer.add_scalar('valid_accuracy', valid_accuracy.item(), self.epoch)
            self.writer.add_scalar('Valid_specificity', valid_specificity.item(), self.epoch)
            self.writer.add_scalar('valid_sensitivity', valid_sensitivity.item(), self.epoch)
            self.writer.add_scalar('Valid_Dice', valid_F1.item(), self.epoch)
            self.writer.add_scalar('Valid_precision', valid_precision.item(), self.epoch)
