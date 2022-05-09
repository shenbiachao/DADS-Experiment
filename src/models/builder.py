# -*- coding: utf-8 -*-

"""Example code for the Builder Module. This code is meant
just for illustrating basic anomaly detection models

Update this when you start working on your own Anomaly Detection project.
"""

import re
import os
import ast
import sys
import glob
import logging

import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------#
#                                 MODULE                                   #
#------------------------------------------------------------------------------#

from .ssad import SSAD
from .supervised import XGB

#------------------------------------------------------------------------------#
#                                 Config                                    #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#                                MODEL BUILDER                                  #
#------------------------------------------------------------------------------#

class BenchmarkBuilder(object):

    """
    Class represents that a builder to build dummy classification model
    """

    @staticmethod
    def build(model_name, config, **kwargs):
        """
        Build a XLMR multi-modal late concat model.

        Arguments:
        ---------
            model_name: model to build

        Returns
        -------
            model: benchmark model
                

        """

        if model_name == 'ssad':
            model = SSAD(config)
        elif model_name == "supervised":
            seed = kwargs.get('seed')
            model  = XGB(config, seed=seed)

        return model
