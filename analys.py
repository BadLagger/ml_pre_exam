import os
import logging as log
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

log.getLogger("PIL").setLevel(log.CRITICAL)


class LamodaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.__df = df.reset_index(drop=True)
        self.__transform = transform
        self.__cache = {}

    def __len__(self):
        return len(self.__df)

    def __getitem__(self, i):
        #log.debug("i = %d (%s)" % (i, len(self.__df)))
        if i in self.__cache:
            image = self.__cache[i]
        else:
            row = self.__df.iloc[i]
            image = Image.open(row['fname']).convert('RGB') 
            self.__cache[i] = image
        
        if self.__transform:
            image = self.__transform(image)
        
        return image, row['label']

class AnalysData:
    def __init__(self, dir):
        self.__dir = dir
        self.__image_data = []
        self.__total_info = {}
    
    def ClassifyFilesBy(self, class_list):
        log.info('Try to classify files in %s' % self.__dir)
        self.__image_data = []
        self.__total_info = {}
        self.__total_info['total_count'] = 0
        self.__total_info['not_jpg'] = 0
        for cl in class_list:
            self.__total_info[cl] = 0
        self.__total_info['witout_class'] = 0
        for img_fname in os.listdir(self.__dir):
            self.__total_info['total_count'] += 1
            #log.debug("Fname: %s" % img_fname)
            if not ".jpg" in img_fname:
                log.error("Not JPG file detected: %s" % img_fname)
                self.__total_info['not_jpg'] += 1
                continue
            current_image_info = {}
            not_class=False
            for cl in class_list:
                if cl in img_fname:
                    self.__total_info[cl] += 1
                    current_image_info['class'] = cl
                    current_image_info['fname'] = img_fname
                    with Image.open("%s/%s" % (self.__dir, img_fname)) as img:
                        width, height = img.size
                    current_image_info['width'] = width
                    current_image_info['height'] = height
                    self.__image_data.append(current_image_info)
                    not_class=True
                    break
            if not_class == False:
                self.__total_info['witout_class'] += 1
                log.error("Not classified file: %s" % img_fname)
        log.info('Classification done!')
        return self.__total_info
    
    def makeBackup(self, filename, backup_dir):
        try:
            if os.path.exists("%s/%s" % (backup_dir, filename)):
                log.debug("Backup for %s already exists" % filename)
                return True
            with open("%s/%s" % (self.__dir, filename), 'rb') as src_f:
                with open("%s/%s" % (backup_dir, filename), "wb") as dst_f:
                    dst_f.write(src_f.read())
        except Exception as expt:
            log.error("Can't create backup for file %s - %s" % (filename, expt))
            return False
        return True
    
    def RestoreFromBackup(self, backup_dir):
        if not os.path.exists(backup_dir):
            log.error("Backup folder is not exists")
            return False
        
        try:
            for img in os.listdir(backup_dir):
                with open("%s/%s" % (backup_dir, img), 'rb') as src_f:
                    with open("%s/%s" % (self.__dir, img), 'wb') as dst_f:
                        dst_f.write(src_f.read())
        except Exception as expt:
            log.error("Can't restore from backup files: %s" % expt)
            return False
        
        return True
            
            
    def ChangeResolutionTo(self, width, height, backup_dir):
        
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        
        for img in self.__image_data:
            if width != img['width'] and height != img['height']:
                log.info("Wrong resolution for file: %s" % img['fname'])
                if self.makeBackup(img['fname'], backup_dir) == False:
                    return False
                
                try:
                    with Image.open("%s/%s" % (self.__dir, img['fname'])) as chage_img:
                        chage_img = chage_img.resize((width, height), Image.LANCZOS)
                        if chage_img.mode == 'RGBA':
                            chage_img = chage_img.convert('RGB')
                        chage_img.save("%s/%s" % (self.__dir, img['fname']))
                except Exception as exp:
                    log.error("Can't change size for image: %s - %s" % (img['fname'], exp))
                    return False
                    
        return True
    
    def GetDataframe(self):
        return pd.DataFrame(self.__image_data)
    
    def GetDataframeForPyTorch(self):
        res_df =pd.DataFrame(self.__image_data)
        res_df['fname'] = res_df['fname'].apply(lambda x: os.path.join(self.__dir, x))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(res_df['class'].unique())}
        res_df['label'] = res_df['class'].map(class_to_idx)
        res_df = res_df.drop(columns=['class', 'width', 'height'])
        return res_df, class_to_idx

class RSNAModel(nn.Module):
    def __init__(self, num_classes=2):
        super(RSNAModel, self).__init__()
        
        # Размеры после каждого слоя (пример для входного размера 66x46)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [32, 66, 46]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [32, 33, 23]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [64, 33, 23]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [64, 16, 11]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [128, 16, 11]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [128, 8, 5]
        )
        
        # Автоматический расчет размера для полносвязного слоя
        self.fc_input_size = self._get_fc_input_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_fc_input_size(self):
        # Пробный проход для определения размера
        x = torch.zeros(1, 3, 66, 46)  # Ваш размер изображения
        x = self.features(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Выравниваем с учетом батча
        x = self.classifier(x)
        return x

def train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch, device, log_wandb=True, verbose=False):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    
    progress_bar = tqdm(train_dataloader, desc=f'Train Epoch {epoch}', disable=not verbose)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        targets.extend(target.cpu().numpy())
        
        # Обновление progress bar
        progress_bar.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = accuracy_score(targets, preds)
    
    if verbose:
        print(f'Train Epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def valid_one_epoch(model, valid_dataloader, loss_fn, epoch, device, log_wandb=True, verbose=False):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    
    progress_bar = tqdm(valid_dataloader, desc=f'Valid Epoch {epoch}', disable=not verbose)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = loss_fn(output, target)
        
        running_loss += loss.item()
        preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(valid_dataloader)
    epoch_acc = accuracy_score(targets, preds)
    
    if verbose:
        print(f'Valid Epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc
        