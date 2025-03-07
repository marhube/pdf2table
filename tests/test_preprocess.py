import os
import sys
import re
import math
#
import pandas as pd
from PIL import ImageFile
from pdf2image import convert_from_path
#
import pdf2table.preprocess as preprocess
from pdf2table.preprocess import ParsePDF,convert_pdf_to_images_in_memory
#
import unittest

test_data_dir = "/home/m01315/General_Python/Packages/pdf2table/test_data"
path_pdf_file = os.path.join(test_data_dir,'Nordre_Land_Eiendomsskatteliste+2024.pdf')
path_cleaned_pdf_file = os.path.join(test_data_dir,'Nordre_Land_processed_pdf_page_1.pdf')
#
filenames_background_snapshots = [filename for filename in os.listdir(test_data_dir) if filename.startswith("Nordre_Land") and filename.endswith(".png")]
path_background_snapshot = [os.path.join(test_data_dir,filename) for filename in  filenames_background_snapshots]
#
image_pymupdf = convert_pdf_to_images_in_memory(path_pdf_file, first_page=1, last_page=1,zoom=7)[0] 
image_tesseract = convert_from_path(path_pdf_file,dpi = 300,first_page=1,last_page=1)[0]
#
parse_pdf_obj_pymupdf = ParsePDF(image = image_pymupdf,path_background_snapshot = path_background_snapshot)
parse_pdf_obj_tesseract = ParsePDF(image = image_tesseract,path_background_snapshot = path_background_snapshot)
#
class TestPreprocess(unittest.TestCase):
    def test_convert_pdf_to_images_in_memory(self):
        print("Er nå inne i test_convert_pdf_to_images_in_memory")
        self.assertIsInstance(image_pymupdf,ImageFile.ImageFile)
    #
    def test_instantiation(self):
        print("Er nå inne i TestPreprocess.test_instantiation")
        subresults = [] 
        #Tester nå både med "pymupdf (fitz)-image" og "tesseract-image"
        subresults.append((isinstance(parse_pdf_obj_pymupdf,ParsePDF)))
        subresults.append((isinstance(parse_pdf_obj_tesseract,ParsePDF)))
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_calc_pixel_color(self):
        print("Er nå inne i TestPreprocess.test_calc_pixel_color")
        subresults = []
        for png_path in path_background_snapshot:
            comparison_value = (149,179,215)
            if re.search(pattern=f"Nordre_Land.*overskrift.*\.png",string = os.path.basename(png_path)) is not None:
                comparison_value = (141,179,226)
            #
            pixel_color = ParsePDF.calc_pixel_color(png_path)
            subresults.append((pixel_color == comparison_value))
        #    
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_remove_background_color(self):
        print("Er nå inne i TestPreprocess.test_remove_background_color")
        image_is_filtered = True
        pixel_values_to_be_filtered = [(149,179,215),(141,179,226)]
        image = parse_pdf_obj_tesseract.image.copy()
        pixels = image.load()
        #
        for pixel_value in pixel_values_to_be_filtered:
            for i in range(image.width):
                for j in range(image.height): 
                    r, g, b = pixels[i, j]
                    # Calculate Euclidean distance between the current pixel and the target color
                    distance = math.sqrt((r - pixel_value[0]) ** 2 + 
                                         (g - pixel_value[1]) ** 2 + 
                                         (b - pixel_value[2]) ** 2)
                    # Replace pixel with white if distance is within threshold
                    if distance < parse_pdf_obj_tesseract.image_filter_threshold:
                        image_is_filtered = False
                        break
                    #
                #
            #
        #
        self.assertTrue(image_is_filtered)
        
    #
    def test_preprocess_image(self):
        print("Er nå inne i TestPreprocess.test_preprocess_image")
        subresults = []
        image_is_filtered = True
        pixel_values_to_be_filtered = [(149,179,215),(141,179,226)]
        image = parse_pdf_obj_tesseract.image.copy()
        pixels = image.load()
        #
        for pixel_value in pixel_values_to_be_filtered:
            for i in range(image.width):
                for j in range(image.height): 
                    r, g, b = pixels[i, j]
                    # Calculate Euclidean distance between the current pixel and the target color
                    distance = math.sqrt((r - pixel_value[0]) ** 2 + 
                                         (g - pixel_value[1]) ** 2 + 
                                         (b - pixel_value[2]) ** 2)
                    # Replace pixel with white if distance is within threshold
                    if distance < parse_pdf_obj_tesseract.image_filter_threshold:
                        image_is_filtered = False
                        break
                    #
                #
            #
        #
        subresults.append(image_is_filtered)
        self.assertTrue(pd.Series(subresults).all())
