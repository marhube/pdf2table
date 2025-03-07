import io
import sys
import math
#
import pandas as pd
import numpy as np
import sys
#fitz kommer fra PyMuPDF
import fitz  # type: ignore  
import PIL
from PIL import Image,ImageFile
#
from typing import Tuple,Self
# 
from pdf2table.grid import CustomizedException
#
def raise_wrong_datatype_pixel_color_exception(pixel_color) -> None:
    raise CustomizedException(f"pixel_color is of class {pixel_color.__class__}  instead of tuple[int,int,int].") 
#
def raise_image_failed_to_load_exception() -> None:
    raise CustomizedException(f"Image failed to load") 
#
def validate_pixel_color(pixel_color):
    is_correct_format = False
    if isinstance(pixel_color,tuple) and len(pixel_color) == 3:
        boolean_list = [isinstance(value,int) for value in pixel_color]
        is_correct_format = pd.Series(boolean_list).all()
    #
    if not is_correct_format:
        raise_wrong_datatype_pixel_color_exception(pixel_color)
    #
#
def convert_pdf_to_images_in_memory(pdf_path : str, first_page : int=1, last_page : int|None =None, zoom : int =2)-> list[ImageFile.ImageFile]:
    """
    Convert specified PDF pages to images in memory using PyMuPDF.

    Args:
        pdf_path (str): Path to the input PDF file.
        first_page (int): The first page to process (1-based index).
        last_page (int): The last page to process (1-based index, inclusive). Defaults to the last page of the PDF.
        zoom (int): Zoom factor to control the resolution of the output images.

    Returns:
        list of PIL.Image.Image: A list of images representing the PDF pages.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)

    # Ensure page numbers are within bounds
    if first_page < 1 or first_page > total_pages:
        raise ValueError("First page number is out of range.")
    if last_page is None:
        last_page = total_pages
    elif last_page < first_page or last_page > total_pages:
        raise ValueError("Last page number is out of range.")

    images = []

    # Convert each page to an image
    for page_number in range(first_page - 1, last_page):
        page = pdf_document[page_number]
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)
        image_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)

    pdf_document.close()
    return images
#
class ParsePDF:
    def __init__(
        self,
        image: PIL.Image.Image,
        path_background_snapshot: str|list[str] = [],
        image_filter_threshold : int = 10
        ) :
          # Initialiserer verdier
          self.image =  image
          self.path_background_snapshot = path_background_snapshot
          self.image_filter_threshold = image_filter_threshold
          self.preprocess_image()
        #
    #
    @staticmethod
    def calc_pixel_color(image_path: str) -> float|tuple[int, ...]|None:
        image = Image.open(image_path)
        # Get the color of a representative pixel in the blue area
        # Assuming the blue is in the top section of the image based on the user's description
        # We'll sample the color from a pixel in that areaQ
        # Get the color of a representative pixel in the blue area
        # Assuming the blue is in the top section of the image based on the user's description
        # We'll sample the color from a pixel in that area
        # Sample color from the middle of the blue line
        mid_x_pos = int(np.round(image.width/2))
        mid_y_pos = int(np.round(image.height/2))
        pixel_color = image.getpixel((mid_x_pos, mid_y_pos))  # Coordinates within the blue area (adjusted as necessary) 
        validate_pixel_color(pixel_color)    
        return pixel_color  
    #
    def remove_background_color(self,rgb_value: tuple)-> Self:
        validate_pixel_color(rgb_value)
        """
        Removes pixels close to a specified background color and replaces them with white.

        Parameters:
        - image (PIL.Image): The input image to process.
        - rgb_value (tuple): The RGB value of the background color to remove (e.g., (229, 229, 229)).
        - threshold (int, optional): The Euclidean distance threshold for color matching (default: 10).

        Returns:
        - PIL.Image: The processed image with the background color removed.
        """
        # OBS: annen datatype
        #Help on function convert in module PIL.Image:      
        # Ensure the image is in RGB mod
        #
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        #
        pixels = self.image.load()
        #
        if type(pixels).__name__ == 'PixelAccess' and pixels is not None and not isinstance(pixels,float):
            for i in range(self.image.width):
                for j in range(self.image.height): 
                    r, g, b =  pixels[i,j] # type: ignore
                    pixels[i, j]
                    # Calculate Euclidean distance between the current pixel and the target color
                    distance = math.sqrt((r - rgb_value[0]) ** 2 + 
                                         (g - rgb_value[1]) ** 2 + 
                                         (b - rgb_value[2]) ** 2)
                    # Replace pixel with white if distance is within threshold
                    if distance < self.image_filter_threshold:
                        pixels[i, j] = (255, 255, 255)  # Replace with white
                    #
                #
            #
        else:
            print(f"pixels har av en eller annen grunn fått feil datatype.\nNå er datatypen til pixels {pixels.__class__}")
            raise_image_failed_to_load_exception()                  
        #            
        return self
        #
    def preprocess_image(self) -> Self:
        #Vil ha filstier som liste selv om det bare er én
        path_backgounds_snapshot_list = self.path_background_snapshot.copy() if isinstance(self.path_background_snapshot,list) else [self.path_background_snapshot]
        #
        for path_to_snapshot in path_backgounds_snapshot_list:
            pixel_color = self.calc_pixel_color(path_to_snapshot)
            if isinstance(pixel_color,tuple):
                self.remove_background_color(pixel_color)
            else:
                raise_wrong_datatype_pixel_color_exception(pixel_color)
        #
        return self
    #   
#
