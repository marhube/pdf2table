import os
import sys
import re
#
import pandas as pd
from pandas import Index # For å gjøre mypy fornøyd
import pyarrow 
import numpy as np
import fitz  # PyMuPDF
import warnings
#
from pdf2table.fitz2miner import Fitz2tesseract
#
import unittest
#
# Start: Lager noe testdata

data_dir = os.path.join("/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/grid_åsnes")
#
#*********** Start fitz"
path_pdf = os.path.join(data_dir,'Skatteliste offentlig ettersyn 2025.pdf')
#
list_fitz_frames = []
doc = fitz.open(path_pdf)
for page in doc:
    words = page.get_text("words")  # Extract words with bbox                   
    # Convert to DataFrame
    fitz_frame = pd.DataFrame(words, columns=['x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'line_no', 'word_no'])
    list_fitz_frames.append(fitz_frame)
#
fitz2miner_obj = Fitz2tesseract(filepath = path_pdf,warn_of_blank_pages = False)
#
class Test_fitz2miner(unittest.TestCase):
    #
    def test_instantiation(self):
        print("Er nå inne i Test_fitz2miner.test_instantiation")
        self.assertIsInstance(fitz2miner_obj,Fitz2tesseract)
    #
    def test_set_bbox_info(self):
        print("Er nå inne i Test_fitz2miner.test_set_bbox_info")
        subresults = [] 
        subresults.append(len(fitz2miner_obj.bbox_info) == len(list_fitz_frames)) 
        for enum,bbox_info_frame in enumerate(fitz2miner_obj.bbox_info):
            comparison_frame = list_fitz_frames[enum]
            subresults.append(bbox_info_frame.shape  == comparison_frame.shape)
            for col in bbox_info_frame.columns:
                subresults.append(bbox_info_frame[col].to_list() == comparison_frame[col].to_list())
            #
        #                
        self.assertTrue(pd.Series(subresults).all()) 
        #
    #
    def test_initialize_tesseract_frame(self):
        print("Er nå inne i Test_fitz2miner.test_initialize_tesseract_frame")
        initial_frame = fitz2miner_obj.initialize_tesseract_frame(pd.DataFrame())
        comparison_frame = pd.DataFrame()
        comparison_frame['text'] = ""
        for col in ['conf','top','bottom','left','right']:
            comparison_frame[col] = pd.array([],dtype=pd.Int64Dtype())
        #
        subresults = []
        #Memo til selv: Sjekker at det er samme datatype
        for col in comparison_frame:
            subresults.append(str(initial_frame[col].dtype) == str(comparison_frame[col].dtype)  )
            subresults.append(initial_frame[col].to_list() == comparison_frame[col].to_list())
        # 
        self.assertTrue(pd.Series(subresults).all()) 
    #
#
