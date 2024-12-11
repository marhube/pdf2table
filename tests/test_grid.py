import os
import sys
import re
#
import pandas as pd
import pyarrow 
import numpy as np

#**********

#Forsøk på  systematisk testing
from pdf2table.grid import GridMiner

import unittest
#
test_data_dir = "/home/m01315/General_Python/Packages/pdf2table/test_data"
path_page_data =  os.path.join(test_data_dir,'tesseract_data.parquet')
page_data = pd.read_parquet(path_page_data,engine="pyarrow")
test_values = pd.read_csv(os.path.join(test_data_dir,"test_values.csv"),header=0,sep=";")
grid_miner_obj = GridMiner(tesseract_page_data = page_data,skip_lines_top=1,skip_lines_bottom=1,hard_coded_regex_filter = [r'^0\S$'] )
column_names = ['Adresse','Eiendom','Takst','Skattenivå','Bunnfradrag','Grunnlag','Promillesats','Skatt','Fritak']
#

class TestGrid(unittest.TestCase):
    def test_instantiation(self):
        self.assertIsInstance(grid_miner_obj,GridMiner)
    #
    def test_set_right(self):
        print("Er nå inne i test_set_right")  
        subresults = []    
        for row in grid_miner_obj.tesseract_page_data.itertuples():
            subresults.append((getattr(row,"right") == getattr(row,"left") + getattr(row,"width")))
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_set_bottom(self):
        print("Er nå inne i test_set_bottom")
        subresults = []    
        for row in grid_miner_obj.tesseract_page_data.itertuples():
            subresults.append((getattr(row,"bottom") == getattr(row,"top") + getattr(row,"height")))
        #
        self.assertTrue(pd.Series(subresults).all()) 


    #
    def test_find_row_boundary_candidates(self):
        print("Er nå inne i test_find_row_boundary_candidates")
        row_boundary_candidates =  grid_miner_obj.find_row_boundary_candidates()
        page_data= grid_miner_obj.tesseract_page_data.query("conf > 0").copy()
        is_increasing = True
        prev_value =  -1000
        for row_boundary_candidate in row_boundary_candidates:
            if row_boundary_candidate <= prev_value or (
                page_data.query(f"top <= {row_boundary_candidate} and bottom >= {row_boundary_candidate}").shape[0] > 0):
                is_increasing = False
                break
            #
        #
        self.assertTrue(is_increasing)
    #
    #Memo: Skal teste at "vertikale skillelinjer" er stigende
    def test_extract_row_boundaries(self):
        print("Er nå inne i test_extract_row_boundaries")
        row_boundaries =  grid_miner_obj.extract_row_boundaries().sort_values(by='top_boundary')
        is_increasing = True
        prev_value =  -1000
        for idx,row in row_boundaries.iterrows():
            if row['top_boundary'] < prev_value or row['top_boundary'] >= row['bottom_boundary'] or (
                page_data.query(f"conf > 0 and top <= {row['top_boundary']} and bottom >= {row['bottom_boundary']}").shape[0] > 0):
                is_increasing = False
                break
            #
        #
        self.assertTrue(is_increasing)
    #
    #Tester foreløpig bare at kolonnnenavnene er på samme rad
    def test_set_rownum(self):
        print("Er nå inne i test_set_rownum")
        page_data_with_rownum = grid_miner_obj.tesseract_page_data.query("conf > 0").copy()
        headers = [header.strip() for header in page_data_with_rownum.query(f" rownum == {grid_miner_obj.skip_lines_top}")['text']]
        header_comparison = ['Adresse','Eiendom','Takst','Skattenivå','Bunnfradrag','Grunnlag','Promillesats','Skatt','Fritak']
        subresults = []
        #Tester at kolonnenavnen er unike og at det er like mange av dem som det skal være
        subresults.append((len(headers) == len(set(headers))))
        subresults.append((len(headers) == len(header_comparison)))
        equal_values = True
        for header in headers:
            if header.strip() not in header_comparison:
                equal_values = False
            #
        #
        subresults.append(equal_values)
        #   
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_extract_table_area(self):
        print("Er nå inne i test_extract_table_area")
        #Memo til self: Metoden "extract_table_area" starter med de original "tesseract-dataene", og har derfor ingen inputargumenter
        table_area = grid_miner_obj.extract_table_area()
        #Øverste rad skal inneholde kolonnenavn 
        first_rownum = table_area['rownum'].min()
        last_rownum = table_area['rownum'].max()
        top_row = table_area.query(f"conf > 0 and rownum == {first_rownum}").sort_values('left')['text'].map(lambda x: x.strip()).to_list()
        last_row = table_area.query(f"conf > 0 and rownum == {last_rownum}").sort_values('left')['text'].map(lambda x: x.strip()).to_list()
        subresults = []
        subresults.append((top_row == column_names))
        subresults.append((test_values['Eiendom'].to_list()[-1] in last_row))
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_reset_rownum(self):
        print("Er nå inne i test_reset_rownum")
        table_area = grid_miner_obj.extract_table_area()
        table_area_new_rownum_values = grid_miner_obj.reset_rownum(table_area)
        sorted_rownum = sorted(table_area_new_rownum_values['rownum'].unique())
        self.assertEqual(sorted_rownum, list(range(len(sorted_rownum))))
    #
    def test_extract_header_area(self):
        print("Er nå inne i test_extract_header_area")
        header_area = grid_miner_obj.extract_header_area()
        header_text = [value.strip() for value in  header_area['text'].to_list()]
        self.assertEqual(header_text, column_names)
    #
    #
    def test_extract_column_boundaries(self):
        print("Er nå inne i test_extract_column_boundaries")
        column_boundaries =  grid_miner_obj.extract_column_boundaries().sort_values(by="left_boundary")
        colnum_values = column_boundaries['colnum'].to_list()
        column_name_values = column_boundaries['column'].to_list()        
        table_area = grid_miner_obj.extract_table_area().query("conf > 0")
        filtered_page_data = grid_miner_obj.filter_column_boundary_data(table_area)
        no_overlap = True
        for row in column_boundaries.itertuples():
            for boundary_col in ['left_boundary','right_boundary']:
                unexpected_rows = filtered_page_data.query(f"left <= {getattr(row,boundary_col)} and right >= {getattr(row,boundary_col)}") 
                if  unexpected_rows.shape[0] > 0:
                    no_overlap  = False
                    break
                #
            #
        #
        subresults = []
        subresults.append((colnum_values == list(range(len(column_names)))))
        subresults.append((column_name_values == column_names))
        subresults.append(no_overlap)
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_add_colnum(self):
        print("Er nå inne i test_add_colnum")
        page_data_with_colnum = grid_miner_obj.add_colnum().query("conf > 0").sort_values(["rownum","left"])
        header_values = [value.strip() for value in  page_data_with_colnum.query("rownum == 0")['text'].to_list()]
        table_content = page_data_with_colnum.query("conf > 0 and rownum > 0")
        subresults = []
        subresults.append((header_values == column_names))
        cadastre_values = [value.strip() for value in table_content.query(f"column == 'Eiendom'")['text']]
        #Sjekker at matrikkeladressene er de samme
        subresults.append((cadastre_values == test_values['Eiendom'].to_list()))
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_extract_column_values(self):
        print("Er nå inne i test_extract_column_values")
        page_table_with_colnum = grid_miner_obj.add_colnum()
        subresults = []
        for column in column_names:
            raw_column_values = grid_miner_obj.extract_column_values(page_table_with_colnum,column)
            percent_pattern = r"(%|‰).*"
            column_values = [re.sub(pattern=percent_pattern,repl='',string= value) for value in raw_column_values]
            #OBSSSSSSSSS Gjør nå en "spesialfiltrering"
            raw_comparison_values =  [value for value in test_values[f"{column}"]]
            if column in  ['Skatt',"Fritak"]:
                # Remove the element at index 1 (second element)
                del raw_comparison_values[1]
                del column_values[1]
                #
            #                                            
            comparison_values = [re.sub(pattern=percent_pattern,repl='',string= str(value)) for value in raw_comparison_values]
            comparison_result = (column_values == comparison_values)
            if not comparison_result:
                print(f"column er {column},len(column_values) er {len(column_values)} column_values er\n{column_values} og comparison_values er {comparison_values}")     
                sys.exit()
            subresults.append((column_values == comparison_values))
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_extract_table(self) -> pd.DataFrame:
        print("Er nå inne i test_extract_table")
        values_df = grid_miner_obj.extract_table()
        print(values_df.shape)
        #Tester her bare dimensjon og kolonnenavn
        subresults = []
        subresults.append((values_df.shape == test_values.shape))
        subresults.append(list(values_df.columns) == list(test_values.columns))
        #Sjekker at alle verdiene er strenger
        all_strings = True
        for row in values_df.itertuples():
            for col in values_df.columns:
                if not isinstance(getattr(row,col),str):
                    all_strings = False
                    break
                #
            #
        #
        subresults.append(all_strings)
        #  
        self.assertTrue(pd.Series(subresults).all())
    #
#   
