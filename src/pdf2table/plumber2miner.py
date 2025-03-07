import os
import sys
import time
import re
from math import floor,ceil
#
import pandas as pd
import pdfplumber
import warnings

#
from typing import Self
#Egenutviklet pakke for å lese inn tabell fra pdf på  den standarden som ca halvparten av pdf-ene med kommunale takster følger
from pdf2table.grid import GridMiner,CustomizedException
#
def raise_invalid_filepath(filepath: str) -> None:
    raise CustomizedException(f"The filepath {filepath} is invalid") 
#
def raise_invalid_scaling_factor(scaling_type : str) -> None:
    raise CustomizedException(f"The  value of  {scaling_type} needs to be a positive integer > 0.")
#
#Memo til selv: Bakgrunnen for valget om å ha høyere oppløsning for rader ("vertical_scaling-factor") enn for kolonner ("horizontal_scaling_factor")
# er at jeg har sett at radene i "pdf-tabelelr" typisk står tettere på hverandre enn kolonnene.
class Plumber2tesseract:
    def __init__(
            self,
            filepath : str,
            vertical_scaling_factor : int = 100,
            horizontal_scaling_factor: int  = 10,
            warn_of_blank_pages: bool = True
    ):
        #Sjekker først at filstien peker til en faktisk fil
        if not os.path.isfile(filepath):
            raise_invalid_filepath(filepath)
        elif not (isinstance(vertical_scaling_factor,int) and vertical_scaling_factor > 0):
              raise_invalid_scaling_factor("vertical_scaling_factor")
        elif not (isinstance(horizontal_scaling_factor,int) and horizontal_scaling_factor > 0):
              raise_invalid_scaling_factor("horizontal_scaling_factor")     
        #
        self.filepath = filepath
        self.vertical_scaling_factor = vertical_scaling_factor
        self.horizontal_scaling_factor = horizontal_scaling_factor
        self.warn_of_blank_pages = warn_of_blank_pages
        self.set_bbox_info()
    #
    def set_bbox_info(self) -> Self:
        list_pdfplumber_frames : list[pd.DataFrame] = []
        with pdfplumber.open(self.filepath) as pdf:
            for page in pdf.pages:
                #Trekker ut en liste av dictionaries
                words_on_page_dict_list = page.extract_words()
                pdfplumber_frame = pd.DataFrame(words_on_page_dict_list)
                list_pdfplumber_frames.append(pdfplumber_frame)
            #
        #
        self.bbox_info =  list_pdfplumber_frames
        return self
    #
    #Memo til selv: Hele hensikten til "initialize_tesseract_frame" er å gjøre "create_tesseract_from_pdfplumber" "robust mot blanke pdfsider"
    #
    def initialize_tesseract_frame(self,pdfplumber_frame: pd.DataFrame) -> pd.DataFrame:
        tesseract_frame = pd.DataFrame()
        if pd.Series( [col in list(pdfplumber_frame.columns) for col in ['text','top','bottom','x0','x1']] ).all(): # Sjekker om har alle nødvendige kolonner
            tesseract_frame = pdfplumber_frame[['text']].copy()
            tesseract_frame['conf'] = 100 # Må være et positivt tall
            tesseract_frame['top'] = pdfplumber_frame['top'].map(lambda x: floor(x*self.vertical_scaling_factor))
            tesseract_frame['bottom'] = pdfplumber_frame['bottom'].map(lambda x: ceil(x*self.vertical_scaling_factor))
            tesseract_frame['left'] = pdfplumber_frame['x0'].map(lambda x: floor(x*self.horizontal_scaling_factor))
            tesseract_frame['right'] = pdfplumber_frame['x1'].map(lambda x: ceil(x*self.horizontal_scaling_factor))
        else: #Lager ellers en "skalltabell" uten rader
            if self.warn_of_blank_pages:
                warnings.warn("At least one of the pages of the pdf is blank!", UserWarning)
            #
            tesseract_frame['text'] = ""
            for col in ['conf','top','bottom','left','right']:
                tesseract_frame[col] = pd.array([],dtype=pd.Int64Dtype())
            # 
        #
        return tesseract_frame
    #
    def create_tesseract_from_pdfplumber(self,pdfplumber_frame: pd.DataFrame) -> pd.DataFrame:
        tesseract_frame =  self.initialize_tesseract_frame(pdfplumber_frame)
        tesseract_frame['height'] = tesseract_frame['bottom'] - tesseract_frame['top']
        tesseract_frame['width'] = tesseract_frame['right'] - tesseract_frame['left']
        required_columns = GridMiner.get_required_columns()
        return tesseract_frame[required_columns]
    #
    def convert_from_path(self) -> list[pd.DataFrame]:
        list_tesseract_frames = []
        #        
        for bbox_frame in self.bbox_info:
            #Sjekker om inneholder alle nødvendige felter            
            tesseract_frame = self.create_tesseract_from_pdfplumber(bbox_frame)
            list_tesseract_frames.append(tesseract_frame)
        #
        return list_tesseract_frames
    #
#  
          
3