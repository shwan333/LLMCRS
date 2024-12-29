import sys
sys.path.append("..")

from src.model.KBRD import KBRD
from src.model.BARCOR import BARCOR
from src.model.UNICRS import UNICRS
from src.model.CHATGPT import CHATGPT
from src.model.OPEN_MODEL import OPEN_MODEL
import argparse

name2class = {
    'kbrd': KBRD,
    'barcor': BARCOR,
    'unicrs': UNICRS,
    'chatgpt': CHATGPT,
    'openmodel': OPEN_MODEL,
}

class RECOMMENDER():
    # def __init__(self, crs_model, *args, **kwargs) -> None:
    def __init__(self, args: argparse.Namespace) -> None:
        model_class = name2class[args.crs_model]
        self.crs_model = model_class(args)
        
    def get_rec(self, conv_dict):
        return self.crs_model.get_rec(conv_dict)
    
    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)
    
    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)
    
    def get_batch_rec(self, conv_dict_list):
        return self.crs_model.get_batch_rec(conv_dict_list)

    def get_batch_conv(self, conv_dict):
        return self.crs_model.get_batch_conv(conv_dict)
    
    def get_sample_conv(self, conv_dict):
        return self.crs_model.get_sample_conv(conv_dict)
    
    def get_sample_batch_conv(self, conv_dict):
        return self.crs_model.get_sample_batch_conv(conv_dict)