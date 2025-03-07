import os
import sys
import time
import re
from math import floor,ceil
#
import pandas as pd
# fitz kommer fra PyMuPDF
import fitz  # type: ignore  
import warnings
from typing import Self
#
from pdf2table.plumber2miner import Plumber2tesseract
from pdf2table.plumber2miner import raise_invalid_filepath
#
class Fitz2tesseract(Plumber2tesseract):
    def __init__(
            self,
            filepath : str,
            vertical_scaling_factor : int = 10,
            horizontal_scaling_factor: int  = 10,
            warn_of_blank_pages: bool = True
    ):
        super().__init__(
            filepath = filepath,
            vertical_scaling_factor = vertical_scaling_factor,
            horizontal_scaling_factor =horizontal_scaling_factor,
            warn_of_blank_pages = warn_of_blank_pages
            )
        if not os.path.isfile(filepath):
            raise_invalid_filepath(filepath)
        #
        self.filepath = filepath
        self.vertical_scaling_factor = vertical_scaling_factor
        self.horizontal_scaling_factor = horizontal_scaling_factor
        self.warn_of_blank_pages =  warn_of_blank_pages
        self.set_bbox_info()
    #
    def set_bbox_info(self) -> Self:
        list_fitz_frames : list[pd.DataFrame] = []
        doc = fitz.open(self.filepath)           
        for page in doc:
            words = page.get_text("words")  # Extract words with bbox                   
            # Convert to DataFrame
            fitz_frame = pd.DataFrame(words, columns=['x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'line_no', 'word_no'])
            list_fitz_frames.append(fitz_frame)
        #
        self.bbox_info =  list_fitz_frames
        return self
    #
        #Memo til selv: Hele hensikten til "initialize_tesseract_frame" er å gjøre "create_tesseract_from_fitz" "robust mot blanke pdfsider"
    #
    def initialize_tesseract_frame(self,fitz_frame: pd.DataFrame) -> pd.DataFrame:
        tesseract_frame = pd.DataFrame()
        if pd.Series( [col in list(fitz_frame.columns) for col in ['text','y0','y1','x0','x1']] ).all(): # Sjekker om har alle nødvendige kolonner
            tesseract_frame = fitz_frame[['text']].copy()
            tesseract_frame['conf'] = 100 # Må være et positivt tall
            tesseract_frame['top'] = fitz_frame['y0'].map(lambda x: floor(x*self.vertical_scaling_factor))
            tesseract_frame['bottom'] = fitz_frame['y1'].map(lambda x: ceil(x*self.vertical_scaling_factor))
            tesseract_frame['left'] = fitz_frame['x0'].map(lambda x: floor(x*self.horizontal_scaling_factor))
            tesseract_frame['right'] = fitz_frame['x1'].map(lambda x: ceil(x*self.horizontal_scaling_factor))
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

#


    