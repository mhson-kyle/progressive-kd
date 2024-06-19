import os
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class TrainManager(object):
    def __init__(self, args, student, teacher=None, ta_list=None, train_loader=None, test_loader=None):
        self.student = student
        self.teacher = teacher
        for i, ta in enumerate(ta_list):
            globals()["self.ta{}".format(i + 1)] = ta

        self.have_teacher = bool(self.teacher)
        self.device = args.device
        self.name = args.experiment_name

        self.teacher.eval()
        self.teacher.train(mode=False)
        for i, ta in enumerate(ta_list):
            globals()["self.ta{}".format(i + 1)].eval()
            globals()["self.ta{}".format(i + 1)].train(mode=False)

        self.train_loader = train_loader

    def train(self):
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        drop_num = self.config['drop_num']

        iteration = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                iteration += 1
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                student_output = self.student(data)

                # Standard Learning Loss (Classification Loss)
                loss_SL = criterion(student_output, target)

                teacher_outputs = self.teacher(data)
                ta_outputs = []
                for i in range(len(ta_list)):
                    ta_outputs.append(globals()["self.ta{}".format(i + 1)](data))

                # Teacher Knowledge Distillation Loss
                loss_KD_list = [nn.KLDivLoss()(F.log_softmax(student_output / T, dim=1),
                                               F.softmax(teacher_outputs / T, dim=1))]

                # Teacher Assistants Knowledge Distillation Loss
                for i in range(len(ta_list)):
                    loss_KD_list.append(nn.KLDivLoss()(F.log_softmax(student_output / T, dim=1),
                                                       F.softmax(ta_outputs[i] / T, dim=1)))

                # Stochastic DGKD
                if args.drop_num != 0:
                    for _ in range(args.drop_num):
                        loss_KD_list.remove(random.choice(loss_KD_list))

                # Total Loss
                loss = (1 - lambda_) * loss_SL + lambda_ * T * T * sum(loss_KD_list)

                loss.backward()
                self.optimizer.step()

            print("epoch {}/{}".format(epoch, epochs))

            self.save(epoch, name='DGKD_{}_{}_best.pth.tar'.format(args.gpus, self.name, args.dataset))
            print('loss: ', loss.data)
            print()

        return best_acc

    def save(self, epoch, name=None):
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, name)

