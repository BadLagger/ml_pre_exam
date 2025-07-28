import sys
import os
import logging as log
from analys import AnalysData, RSNAModel, get_result_from_test
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

ARGS_NUM=4

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
        
        return image

'''
    Первый аргумент - путь к файлу с моделью
    Второй аргумент - путь к папке с файлами картинок
    Третий аргумент - путь к файлу с классами
'''
if __name__ == "__main__":
    log.basicConfig(
        level=log.INFO,           # Уровень регистрации сообщений
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'    # Формат даты и времени
    )

    log.info("Checking model!")


    if len(sys.argv) != ARGS_NUM:
        log.error("Wrong args number. Should be %s" % ARGS_NUM)
        sys.exit(0)
    
    if not os.path.isfile(sys.argv[1]):
        log.error("First arg should be a file")
        sys.exit(0)

    if not os.path.isdir(sys.argv[2]):
        log.error("Second arg should be a directory")
        sys.exit(0)

    if not os.path.isfile(sys.argv[3]):
        log.error("Third arg should be a file")
        sys.exit(0)

    ad = AnalysData(sys.argv[2])
    info = ad.justLoadFiles()
    log.info(info)

    df = ad.GetDataframe()
    log.info("Resolution statistic:\n%s" % (df[['width', 'height']].describe()))
    log.info("Resolutions width: %s" % df['width'].value_counts())
    log.info("Resolutions height: %s" % df['height'].value_counts())
    log.info("Resolutions width percents: %s" % (df['width'].value_counts(normalize=True) * 100))
    log.info("Resolutions height precents: %s" % (df['height'].value_counts(normalize=True) * 100))
    log.info("Try to change resolution:")
    if ad.ChangeResolutionTo(46, 66, "backups_test") == False:
        log.error("Can't change resolution!")
        sys.exit(0)
    
    test_df = ad.GetDataframeForModel()
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = LamodaDataset(test_df, transform=val_transform)
    pred_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSNAModel(num_classes=2).to(device)

    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in pred_loader:
            images = batch.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    test_df['label'] = predictions

    with open(sys.argv[3], 'r') as class_idx_f:
        class_idx = json.load(class_idx_f)

    result_df = get_result_from_test(test_df, class_idx)

    log.info("Result:")
    log.info(result_df)

    result_df.to_csv("submission.csv", index=False)

    log.info("End!!!")