import os
import logging as log
from PIL import Image
import pandas as pd

log.getLogger("PIL").setLevel(log.CRITICAL)

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
    
    def GetDataframe(self):
        return pd.DataFrame(self.__image_data)