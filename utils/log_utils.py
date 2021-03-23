# -*- coding: utf-8 -*-
import logging
import os
import sys
import datetime


def init_logger(log_path, name='dispnet'):
    # 创建logger
    root = logging.getLogger()
    # 设置日志级别
    root.setLevel(logging.NOTSET)

    fmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'
    formatter = logging.Formatter(fmt)

    # 保存的日志文件
    logfile = os.path.join(log_path, '%s-%s.log' % (name, datetime.datetime.today()))
    # 创建一个handler，用于写入日志文件
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    root.addHandler(fileHandler)

    # 创建一个handler，用于输出到控制台
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    # consoleHandler.terminator = '\n'
    root.addHandler(consoleHandler)

    logging.debug('Logging to %s' % logfile)