import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import os
import datetime

from TIMNET import TIMNET
import pdb

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels
    
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
class TIMNET_Model(nn.Module):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__()
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        
    def train(self, x, y):
        
        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i=1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.model = nn.DataParallel(self.model)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):
            train_x, train_y = x[train_ids], y[train_ids]
            test_x, test_y = x[test_ids], y[test_ids]

            model = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                datashape = self.data_shape,
                                num_classes = self.num_classes,
                                return_sequences=True)
            
            model = model.to(device)

            loss_fn = nn.CrossEntropyLoss().cuda()
            optimizer = Adam(model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=1e-8)

            folder_address = filepath + self.args.data + "_" + str(self.args.random_seed) + "_" + now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".pth"

            train_dataset = MyDataset(train_x, train_y)
            test_dataset = MyDataset(test_x, test_y)

            train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_x), shuffle=False)

            best_acc = 0.0
            
            for epoch in range(self.args.epoch):                
                model.train()
                epoch_train_loss = 0  # 각 에폭마다의 총 손실을 저장하기 위한 변수
                epoch_train_acc = 0  # 각 에폭마다의 총 맞춘 예측 수를 저장하기 위한 변수


                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    pred = model.forward(data.cuda())
                    
                    loss = loss_fn.forward(pred, target)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    predicted = torch.argmax(pred, dim=1)
                    # target에서 1의 위치를 찾아 실제 클래스를 얻습니다.
                    actual = torch.argmax(target, dim=1)
                    # 예측된 클래스와 실제 클래스가 일치하는지 확인하고, 그 수를 계산합니다.
                    correct = (predicted == actual).sum().item()

                    # 일치하는 수를 배치 크기로 나누어 정확도를 계산합니다.
                    acc = correct / target.size(0)
                    epoch_train_acc += acc
                    
                epoch_train_acc = epoch_train_acc / len(train_loader)  # 에폭 정확도 계산
                epoch_train_loss = epoch_train_loss / len(train_loader)
                print(f"Epoch: {epoch+1}/{self.args.epoch},Train Loss: {epoch_train_loss},Train Accuracy: {epoch_train_acc}")  # 에폭 손실과 정확도 출력
                

                model.eval()                
                
                epoch_val_loss = 0
                epoch_val_acc = 0

                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        pred = model.forward(data.cuda())
                        loss = loss_fn.forward(pred, target)

                        epoch_val_loss += loss.item()
                        predicted = torch.argmax(pred, dim=1)
                        actual = torch.argmax(target, dim=1)
                        correct = (predicted == actual).sum().item()
                        acc = correct / target.size(0)

                        epoch_val_acc += acc

                epoch_val_acc = epoch_val_acc / len(test_loader)
                epoch_val_loss = epoch_val_loss / len(test_loader)
                print(f"Epoch: {epoch+1}/{self.args.epoch},Val Loss: {epoch_val_loss},Val Accuracy: {epoch_val_acc}")
                if epoch_val_acc > best_acc:
                        best_acc = epoch_val_acc
                        torch.save(model.state_dict(), weight_path)
                
            
            model.load_state_dict(torch.load(weight_path))
            model.eval()
            total_loss = 0
            total_acc = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model.forward(data.cuda())
                    loss = loss_fn.forward(pred, target)

                    total_loss += loss.item()
                    predicted = torch.argmax(pred, dim=1)
                    actual = torch.argmax(target, dim=1)
                    correct = (predicted == actual).sum().item()
                    acc = correct / target.size(0)

                    total_acc += acc

            total_acc = total_acc / len(test_loader)
            total_loss = total_loss / len(test_loader)
            print(f"Split: {i},Total Loss: {total_loss},Total Accuracy: {total_acc}")
            i += 1

            avg_accuracy += total_acc
            self.matrix.append(confusion_matrix(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy()))
            report = classification_report(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy(), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(report)
            print(classification_report(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy(), target_names=self.class_label))
        print("Average ACC:",avg_accuracy/self.args.split_fold)
                

    def test(self, x, y, path):
        i = 1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for fold, (train_ids, test_ids) in enumerate(kfold.split(x, y)):
            test_x, test_y = x[test_ids], y[test_ids]
            test_dataset = MyDataset(test_x, test_y)
            test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_x), shuffle=False)

            model = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                datashape = self.data_shape,
                                num_classes = self.num_classes,
                                return_sequences=True)
            
            model = model.to(device)

            loss_fn = nn.CrossEntropyLoss().cuda()

            weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".pth"
            model.load_state_dict(torch.load(weight_path))
            model.eval()
            total_acc = 0
            total_loss = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model.forward(data.cuda())
                    loss = loss_fn.forward(pred, target)

                    total_loss += loss.item()
                    predicted = torch.argmax(pred, dim=1)
                    actual = torch.argmax(target, dim=1)
                    correct = (predicted == actual).sum().item()
                    acc = correct / target.size(0)

                    total_acc += acc

            total_acc = total_acc / len(test_loader)
            total_loss = total_loss / len(test_loader)
            print(f"Total Loss: {total_loss},Total Accuracy: {total_acc}")
            i += 1

            avg_accuracy += total_acc
            self.matrix.append(confusion_matrix(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy()))
            report = classification_report(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy(), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(report)
            print(classification_report(actual.detach().cpu().numpy(), predicted.detach().cpu().numpy(), target_names=self.class_label))

        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold
        return x_feats, y_labels
