import sys
import logging as log
import os
from analys import AnalysData, LamodaDataset, RSNAModel, Train
import pandas as pd
from sklearn.model_selection._split import train_test_split
import torchvision
from torchvision import transforms
import torch
import numpy as np
import torch.nn as nn

MIN_ARGS_NUM=2

if __name__ == "__main__":

    log.basicConfig(
        level=log.DEBUG,           # Уровень регистрации сообщений
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'    # Формат даты и времени
    )

    log.info("Start ML lamode")

    if len(sys.argv) < MIN_ARGS_NUM:
        log.error("Args should be greater or equal than: %s" % MIN_ARGS_NUM)
        sys.exit(0)

    if not os.path.isdir(sys.argv[1]):
        log.error("%s is not a dir" % sys.argv[1])
        sys.exit(0)
    
    dataAnalys = AnalysData(sys.argv[1])
    
    if len(sys.argv) == MIN_ARGS_NUM + 1:
        if sys.argv[2] == 'restore':
            if dataAnalys.RestoreFromBackup("backups") == False:
                log.error("Can't restore files!")
                sys.exit(0)     
    
    info = dataAnalys.ClassifyFilesBy(['bryuki', 'bluzy'])

    dataDf = dataAnalys.GetDataframe()
    log.info("Total in df: %s" % len(dataDf))
    log.info("Classes in percents:\n%s" % (dataDf["class"].value_counts(normalize=True) * 100))
    log.info("Resolution statistic:\n%s" % (dataDf[['width', 'height']].describe()))
    log.info("Resolutions width: %s" % dataDf['width'].value_counts())
    log.info("Resolutions height: %s" % dataDf['height'].value_counts())
    log.info("Resolutions width percents: %s" % (dataDf['width'].value_counts(normalize=True) * 100))
    log.info("Resolutions height precents: %s" % (dataDf['height'].value_counts(normalize=True) * 100))
    
    log.info("Try to change resolution:")
    if dataAnalys.ChangeResolutionTo(46, 66, "backups") == False:
        log.error("Can't change resolution!")
        sys.exit(0)
    
    
    pyTorchDf, class_idx = dataAnalys.GetDataframeForPyTorch()
    print(pyTorchDf)
    
    train_df, val_df = train_test_split(pyTorchDf, test_size=0.3)
    
    # Определение трансформаций
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Создание Dataset и DataLoader
    train_dataset = LamodaDataset(train_df, transform=train_transform)
    val_dataset = LamodaDataset(val_df, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Проверка одного батча
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Визуализация
    import matplotlib.pyplot as plt
    
    
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(5)
    
    images, labels = next(iter(train_loader))
    out = torchvision.utils.make_grid(images)
    #imshow(out)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSNAModel(num_classes=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train = Train(model, train_loader, val_loader, optimizer, loss_fn, device)
    for epoch in range(1, 11):  # 10 эпох
        train_loss, train_acc = train.train_one_epoch(epoch, verbose=True)
        val_loss, val_acc = train.valid_one_epoch(epoch, verbose=True)
        train.update_result(epoch, train_loss, train_acc, val_loss, val_acc)
        
    print("Result:\n%s" % train.get_result_df())
    
    # Сохраняем лучшую модель
    model_num, best_model = train.get_best_model()
    print("The best model is: %d" % model_num)
    torch.save(best_model.state_dict(), "best_model.pth")
    
    log.info("End ML lamode")