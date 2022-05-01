# -*- coding: utf-8 -*-
import os
import sys

import toml
import glob
import yaml
import logging
import logging.config
import collections
import configparser
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from os import path as os_path
from sys import path as sys_path

#------------------------------------------------------------------------------#
#                                 LOGGER                                   #
#------------------------------------------------------------------------------#


def setup_log_config(log_path, log_file_name, log_config_file):
    """Set up logger configuration"""

    os.makedirs(log_path, exist_ok=True)
    with open(log_config_file, 'r') as f:
        log_cfg = yaml.safe_load(f.read())
    log_cfg['handlers']['file_handler']['filename'] = os_path.join(
        log_path, log_file_name)
    logging.config.dictConfig(log_cfg)

logger = setup_log_config('./logs/', 'main.log', './log_config.yaml')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)