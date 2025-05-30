import time
import torch
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import cohen_kappa_score


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _new_get_optimizer(self):
        """Returns the optimizer"""
        if self.exemplars_dataset and len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 :
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)
        save_model_path = os.path.join('./model_path','task_'+str(t)+'.pth')
        torch.save(self.model.state_dict(),save_model_path)


    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                #self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                #self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._new_get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc_tar, train_acc_tag,OA_tar, AA_mean_tar, Kappa_tar, AA_tar,OA_tag, AA_mean_tag, Kappa_tag, AA_tag = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAg acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc_tag), end='')
                #self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                #self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc_tar, valid_acc_tag,OA_tar, AA_mean_tar, Kappa_tar, AA_tar,OA_tag, AA_mean_tag, Kappa_tag, AA_tag = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAg acc={:5.1f}%, OA={:5.1f}%,AA={:5.1f}%,Kappa={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc_tag,100*OA_tag,100*AA_mean_tag,100*Kappa_tag), end='')
            #self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            #self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            #self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            #self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            tars = np.array([])
            preds_tar = np.array([])
            preds_tag = np.array([])
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag, pred_tar,pred_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
                tars = np.append(tars, targets.data.cpu().numpy())
                preds_tar = np.append(preds_tar,pred_tar.data.cpu().numpy())
                preds_tag = np.append(preds_tag,pred_tag.data.cpu().numpy())
            OA_tar, AA_mean_tar, Kappa_tar, AA_tar = self.my_cal_oa_aa_ka(tars,preds_tar)
            OA_tag, AA_mean_tag, Kappa_tag, AA_tag = self.my_cal_oa_aa_ka(tars,preds_tag)
            #print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_tar, AA_mean_tar, Kappa_tar))
            #print(AA_tar)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, OA_tar, AA_mean_tar, Kappa_tar, AA_tar,OA_tag, AA_mean_tag, Kappa_tag, AA_tag

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred_taw = torch.zeros_like(targets.to(self.device))
        #pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred_taw)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred_taw[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred_taw == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred_tag = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred_tag = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred_tag == targets.to(self.device)).float()
        return hits_taw, hits_tag, pred_taw, pred_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    
    def cal_oa_aa_ka(self,tar,pre):
        matrix = confusion_matrix(tar, pre)
        shape = np.shape(matrix)
        number = 0
        sum = 0
        AA = np.zeros([shape[0]], dtype=np.float32)
        for i in range(shape[0]):
            number += matrix[i, i]
            if np.sum(matrix[i, :]) == 0:
                AA[i] = 0
            else:
                AA[i] = matrix[i, i] / np.sum(matrix[i, :])
            sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
        OA = number / np.sum(matrix)
        AA_mean = np.mean(AA)
        pe = sum / (np.sum(matrix) ** 2)
        Kappa = (OA - pe) / (1 - pe)
        return OA, AA_mean, Kappa, AA
    def my_cal_oa_aa_ka(self,true,pred):
        Kappa = cohen_kappa_score(np.array(true).reshape(-1, 1), np.array(pred).reshape(-1, 1))
        OA = (pred==true).mean()
        num_label = np.unique(true)
        N = len(num_label)
        avec = []
        for i in range(N):
            class_idx = np.where(true==num_label[i])
            class_pred = pred[class_idx]
            class_true = true[class_idx]
            class_acc = (class_true==class_pred).mean()
            avec.append(class_acc)
        mean_class = np.mean(np.asarray(avec))
        #print('aa: ',mean_class)
        return OA,mean_class,Kappa,avec


