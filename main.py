import sys
import logging as log
import os
from analys import AnalysData
import pandas as pd

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
    info = dataAnalys.ClassifyFilesBy(['bryuki', 'bluzy'])

    dataDf = dataAnalys.GetDataframe()
    log.info("Total in df: %s" % len(dataDf))
    log.info("Classes in percents:\n%s" % (dataDf["class"].value_counts(normalize=True) * 100))
    log.info("Resolution statistic:\n%s" % (dataDf[['width', 'height']].describe()))
    log.info("Resolutions width: %s" % dataDf['width'].value_counts())
    log.info("Resolutions height: %s" % dataDf['height'].value_counts())
    log.info("Resolutions width percents: %s" % (dataDf['width'].value_counts(normalize=True) * 100))
    log.info("Resolutions height precents: %s" % (dataDf['height'].value_counts(normalize=True) * 100))
    log.info("End ML lamode")