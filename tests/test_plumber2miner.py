import os
import sys
import re
#
import pandas as pd
from pandas import Index # For å gjøre mypy fornøyd
import pyarrow 
import numpy as np
import pdfplumber
from math import floor,ceil
import warnings
#
#Egenutviklet pakke for å lese inn tabell fra pdf på  den standarden som ca halvparten av pdf-ene med kommunale takster følger
from pdf2table.grid import GridMiner
from pdf2table.plumber2miner import Plumber2tesseract
#
import unittest
#
# Start: Lager noe testdata


evenes_dir = os.path.join("/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/grid_evenes")
#
plumber_evenes_filepath = os.path.join(evenes_dir,'Eiendomsskatteliste_Evenes_2024.pdf')
# Trekker først ut "pdfplumber"-data på datarammeformat
#
plumber2miner_obj = Plumber2tesseract(filepath = plumber_evenes_filepath,vertical_scaling_factor = 1,horizontal_scaling_factor = 1,warn_of_blank_pages = False)
#
def create_tesseract_from_pdfplumber(pdfplumber_frame: pd.DataFrame,ndigits: int = 0) -> pd.DataFrame:
    tesseract_frame = pdfplumber_frame[['text']].copy()
    tesseract_frame['conf'] = 100 # Må være et positivt tall
    scaling_factor = 10**ndigits
    tesseract_frame['top'] = pdfplumber_frame['top'].map(lambda x: floor(x*scaling_factor))
    tesseract_frame['bottom'] = pdfplumber_frame['bottom'].map(lambda x: ceil(x*scaling_factor)) 
    tesseract_frame['left'] = pdfplumber_frame['x0'].map(lambda x: floor(x*scaling_factor))
    tesseract_frame['right'] = pdfplumber_frame['x1'].map(lambda x: ceil(x*scaling_factor)) 
    tesseract_frame['height'] = tesseract_frame['bottom'] - tesseract_frame['top']
    tesseract_frame['width'] = tesseract_frame['right'] - tesseract_frame['left']
    required_columns = GridMiner.get_required_columns()
    return   tesseract_frame[required_columns]
#
list_pdfplumber_frames = []
list_tesseract_frames = []
#  
with pdfplumber.open(plumber_evenes_filepath) as pdf:
    for page in pdf.pages:
        #Trekker ut en liste av dictionaries
        words_on_page_dict_list = page.extract_words()
        pdfplumber_frame = pd.DataFrame(words_on_page_dict_list)
        tesseract_frame = create_tesseract_from_pdfplumber(pdfplumber_frame)
        list_pdfplumber_frames.append(pdfplumber_frame)
        list_tesseract_frames.append(tesseract_frame)
    #
#  
pdfplumber0 = list_pdfplumber_frames[0]
tesseract0 = list_tesseract_frames[0]
#
class Test_plumber2miner(unittest.TestCase):
    #
    def test_instantiation(self):
        print("Er nå inne i Test_plumber2miner.test_instantiation")
        self.assertIsInstance(plumber2miner_obj,Plumber2tesseract)
    #
    def test_set_bbox_info(self):
        print("Er nå inne i Test_plumber2miner.test_set_bbox_info")
        subresults = [] 
        subresults.append(len(plumber2miner_obj.bbox_info) == len(list_pdfplumber_frames)) 
        for enum,bbox_info_frame in enumerate(plumber2miner_obj.bbox_info):
            comparison_frame = list_pdfplumber_frames[enum]
            subresults.append(bbox_info_frame.shape  == comparison_frame.shape)
            for col in bbox_info_frame.columns:
                subresults.append(bbox_info_frame[col].to_list() == comparison_frame[col].to_list())
            #
        #                
        self.assertTrue(pd.Series(subresults).all()) 
        #
    #
    def test_create_tesseract_from_pdfplumber(self):
        print("Er nå inne i Test_plumber2miner.test_create_tesseract_from_pdfplumber")
        tesseract_data = plumber2miner_obj.create_tesseract_from_pdfplumber(pdfplumber0)
        comparison = tesseract0
        #
        subresults = [] 
        subresults.append(tesseract_data.shape == comparison.shape)
        for col in comparison:
            subresults.append(tesseract_data[col].to_list() == comparison[col].to_list())
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_initialize_tesseract_frame(self):
        print("Er nå inne i Test_plumber2miner.test_initialize_tesseract_frame")
        initial_frame = plumber2miner_obj.initialize_tesseract_frame(pd.DataFrame())
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
    def test_convert_from_path(self):
        print("Er nå inne i Test_plumber2miner.test_convert_from_path")
        list_of_tesseract_data = plumber2miner_obj.convert_from_path()
        list_of_comparison_data = list_tesseract_frames
        #
        subresults = [] 
        subresults.append(len(list_of_tesseract_data) == len(list_of_comparison_data))
        for enum,tesseract_frame in enumerate(list_of_tesseract_data):
            GridMiner.validate_dataframe_columns(tesseract_frame)
            comparison_frame = list_of_comparison_data[enum]
            subresults.append(tesseract_frame.shape == comparison_frame.shape)
            for col in comparison_frame.columns:
                subresults.append(tesseract_frame[col].to_list() == comparison_frame[col].to_list())
            #
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
#