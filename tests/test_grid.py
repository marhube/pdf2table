import os
import sys
import re
from math import floor,ceil
#
import pandas as pd
from pandas import Index # For å gjøre mypy fornøyd
import pyarrow 
import numpy as np
import pandasql as ps 
import warnings

#**********

#Forsøk på  systematisk testing
from pdf2table.grid import GridMiner
from pdf2table.plumber2miner import Plumber2tesseract
#
import unittest
#
test_data_dir = "/home/m01315/General_Python/Packages/pdf2table/test_data"
path_page_data =  os.path.join(test_data_dir,'tesseract_data.parquet')
path_page_data2 =  os.path.join(test_data_dir,'tesseract_data2.parquet')

page_data = pd.read_parquet(path_page_data,engine="pyarrow")
page_data2 = pd.read_parquet(path_page_data2,engine="pyarrow")
test_values = pd.read_csv(os.path.join(test_data_dir,"test_values.csv"),header=0,sep=";")
grid_miner_obj = GridMiner(tesseract_page_data = page_data,skip_lines_top=1,skip_lines_bottom=1)
grid_miner_obj2 = GridMiner(tesseract_page_data = page_data,skip_lines_top=1,skip_lines_bottom= 1)
#
column_names = ['Adresse','Eiendom','Takst','Skattenivå','Bunnfradrag','Grunnlag','Promillesats','Skatt','Fritak']
#
mock_text = "sdfs"
mock_text_list = [f" {mock_text} ",np.nan,"  ",f"{mock_text}",f" {mock_text}",f" {mock_text}"]
mock_pytesseract_data = pd.DataFrame(
    {'text' : mock_text_list,
     'conf' : [56,-1,49,76,45,84],
     'left' : [234,350,78,432,546,235],
     'width' : [70,74,34,64,65,72],
     'top' : [45,46,345,87,89,1200],
     'height' : [23,25,534,22,21,65]     
     }
)
#
mock_grid_miner_obj = GridMiner(tesseract_page_data = mock_pytesseract_data.copy(),skip_lines_top=1,skip_lines_bottom=1)
#
oystre_slidre_dir = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/vertical_span_header"
oystre_slidre_path_parquet = os.path.join(oystre_slidre_dir,"erroneous_rows.parquet" )
pytesseract_data_oystre_slidre = pd.read_parquet(oystre_slidre_path_parquet,engine= "pyarrow")
grid_miner_obj_oystre_slidre = GridMiner(tesseract_page_data = pytesseract_data_oystre_slidre,skip_lines_top = 1,skip_lines_bottom = 1,vertical_span_header = 2)
#
vefsn_dir = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/erroneous_row_numbering_Vefsn"
path_parquet_vefsn = os.path.join(vefsn_dir,"erroneous_rows.parquet" )
pytesseract_data_vefsn = pd.read_parquet(path_parquet_vefsn,engine= "pyarrow")
grid_miner_obj_vefsn = GridMiner(tesseract_page_data = pytesseract_data_vefsn,skip_lines_top=0,skip_lines_bottom=1,vertical_span_header = 2)
#
vefsn_dir = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/erroneous_row_numbering_Vefsn"
path_parquet_vefsn = os.path.join(vefsn_dir,"erroneous_rows.parquet" )
pytesseract_data_vefsn = pd.read_parquet(path_parquet_vefsn,engine= "pyarrow")
pytesseract_data_mock_vefsn = pd.concat([mock_pytesseract_data,pytesseract_data_vefsn])
pytesseract_data_mock_vefsn['bottom'] = pytesseract_data_mock_vefsn['top']  + pytesseract_data_mock_vefsn['height'] 
pytesseract_data_mock_vefsn['right'] = pytesseract_data_mock_vefsn['left']  + pytesseract_data_mock_vefsn['width'] 
#For test  av  situagsjon med "ingen kolonneoverskrifter"
tesseract_dir_no_header = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/grid_leirfjord"
path_parquet_page1 = os.path.join(tesseract_dir_no_header,'pytesseract_data_page1.parquet')
path_parquet_page2 = os.path.join(tesseract_dir_no_header,'pytesseract_data_page2.parquet')
pytesseract_data_page1 = pd.read_parquet(path_parquet_page1,engine= "pyarrow")
pytesseract_data_page2 = pd.read_parquet(path_parquet_page2,engine= "pyarrow")
grid_miner_obj_page1 = GridMiner(tesseract_page_data = pytesseract_data_page1,skip_lines_top=0,skip_lines_bottom=0,vertical_span_header = 1)
header_area_page1 = grid_miner_obj_page1.extract_header_area()
grid_miner_obj_page2 = GridMiner(tesseract_page_data = pytesseract_data_page2,skip_lines_top=0,skip_lines_bottom=0,vertical_span_header = 0,header_area = header_area_page1)
#
tesseract_dir_multiline_column_names = os.path.join("/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/grid_vegårdshei")
path_parquet_multiline = os.path.join(tesseract_dir_multiline_column_names,"page1.parquet")
pytesseract_multiline = pd.read_parquet(path_parquet_multiline,engine= "pyarrow")
grid_miner_obj_multiline = GridMiner(tesseract_page_data = pytesseract_multiline,skip_lines_top=5,skip_lines_bottom=1,vertical_span_header = 1,merge_multiline_column_names = True)
#
multiline_cell_data = pd.DataFrame(
    {
        'text': ['T','a','k','s','t'],
        'left' : [217,222,228,233,237],
        'right' : [223,229,234,238,242],
        'top' : [10874] * 5,
        'bottom' : [11978] * 5,
        'width' : [6,7,6,5,5],
        'height' : [1104] * 5,
        'conf' :  [100] * 5,
        'rownum' : [0] * 5
        })
        #
comparison_dict = {
    'text': 'Takst',
    'left' : 217,
    'right' : 242,
    'top': 10874,
    'bottom': 11978,
    'width' : 25,
    'height' : 1104,
    'rownum' : 0,
    'conf' :  100
    }
    #
#
path_multiline_cell_pdf = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/grid_kvæfjord/Eiendomskatt+2022+-+bolig+og+hytter.pdf"
plumber2miner_multiline_cell_obj = Plumber2tesseract(filepath = path_multiline_cell_pdf,horizontal_scaling_factor = 1,vertical_scaling_factor = 4)
pytesseract_data_multiline_cell = plumber2miner_multiline_cell_obj.convert_from_path()[0]
#
grid_miner_obj_multiline_cell =  GridMiner(
    tesseract_page_data = pytesseract_data_multiline_cell,
    skip_lines_top=2,
    skip_lines_bottom=0,
    vertical_span_header = 3,
    row_grouping_criteria = 2,
    merge_multiline_column_names = True
    )
#
class TestGrid(unittest.TestCase):
    def test_instantiation(self):
        print("Er nå inne i TestGrid.test_instantiation")
        self.assertIsInstance(grid_miner_obj,GridMiner)
    #
    def test_get_required_columns(self):
        print("Er nå inne i TestGrid.test_get_required_columns")
        required_columns = GridMiner.get_required_columns() 
        comparison = ['text','conf','left','width','top','height']
        subresults = [] 
        subresults.append(len(required_columns) == len(comparison))
        for col in comparison:
            subresults.append(col  in  required_columns)
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_set_right(self):
        print("Er nå inne i TestGrid.test_set_right")  
        subresults = []    
        for row in grid_miner_obj.tesseract_page_data.itertuples():
            subresults.append((getattr(row,"right") == getattr(row,"left") + getattr(row,"width")))
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_set_bottom(self):
        print("Er nå inne i TestGrid.test_set_bottom")
        subresults = []    
        for row in grid_miner_obj.tesseract_page_data.itertuples():
            subresults.append((getattr(row,"bottom") == getattr(row,"top") + getattr(row,"height")))         
        self.assertTrue(pd.Series(subresults).all()) 
    #Memo til selv: Sammenligner i test_filter_boundary_data med en krøkkete metode
    #
    def test_remove_bbox_entries(self):
        print("Er nå inne i TestGrid.test_remove_bbox_entries")
        # 
        mock_tesseract_duplicated = pd.concat([mock_grid_miner_obj.tesseract_page_data]*2)
        min_top = 87
        max_top = 500
        bboxes_to_remove = mock_tesseract_duplicated.query(f"conf > 0 and top >= {min_top} and top <={max_top} ")[['top','bottom','left','right']].drop_duplicates()
        mock_tesseract_after_removal = GridMiner.remove_bbox_entries(mock_tesseract_duplicated,bboxes_to_remove)
        comparison_data = mock_tesseract_duplicated.query(f"conf <= 0 or top < {min_top} or top > {max_top}")
        subresults = []
        subresults.append(mock_tesseract_after_removal.shape == comparison_data.shape)
        #
        for col in comparison_data.columns:
            subresults.append(pd.isna(value1) and pd.isna(value2) or value1 == value2 for value1,value2 in  zip(mock_tesseract_after_removal[col].to_list(), comparison_data[col].to_list()))
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_detect_overlapping_bboxes(self):
        print("Er nå inne i TestGrid.test_detect_overlapping_bboxes")
        overlapping_data = grid_miner_obj_vefsn.tesseract_page_data.query("conf > 0").copy()
        overlapping_bboxes = GridMiner.detect_overlapping_bboxes(overlapping_data)
        unique_overlapping_bboxes = overlapping_bboxes[['top','bottom','left','right','text']].sort_values(by=['top','left']).drop_duplicates()
        overlapping_with_first = GridMiner.detect_overlapping_bboxes(left_table = overlapping_data,right_table = unique_overlapping_bboxes.iloc[0:1,:],include_self = True)
        overlapping_with_first['text'] = overlapping_with_first['text'].map(lambda x: re.sub(pattern=r'\W',repl='',string= x))
        overlapping_with_first_manual = pd.DataFrame({'top': [1574,1587],'bottom' : [1628,1620], 'left' : [266,342],'right': [388,378], 'text': ['Chr','nr'] })
        subresults = []
        for col in ['top','bottom','left','right']:
            subresults.append(overlapping_with_first[col].to_list() == overlapping_with_first_manual[col].to_list())
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_filter_overlapping_bboxes(self):
        print("Er nå inne i TestGrid.test_filter_overlapping_bboxes")
        overlapping_data = grid_miner_obj_vefsn.tesseract_page_data.query("conf > 0").copy()
        overlapping_data = grid_miner_obj.filter_overlapping_bboxes(overlapping_data)
        #Skal nå ikke være noen overlappende "bboxes"
        query_overlapping_boxes = """
            SELECT table1.*
            FROM  overlapping_data table1
            INNER JOIN  overlapping_data table2 ON
            (
            (table1.right >= table2.left and table1.left < table2.left) OR
            (table2.right >= table1.left and table2.left < table1.left)
            )  AND
            (
            (table1.bottom >= table2.top and table1.top < table2.top) OR
            (table2.bottom >= table1.top and table2.top < table1.top)
            )        
            """  
        overlapping_boxes =  ps.sqldf(query_overlapping_boxes, locals())
        self.assertEqual(overlapping_boxes.shape[0],0)      
    #
    def test_detect_horizontally_overlapping_bboxes_on_same_row(self):
        print("Er nå inne i TestGrid.test_detect_horizontally_overlapping_bboxes_on_same_row")
        header_area = grid_miner_obj_oystre_slidre.extract_header_area()
        #Gjør ulike tester av funksjonalitetetn som denne funksjonen skal tilby
        subresults = []  
        overlapping_boxes_header =  GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(header_area,include_self=False)
        subresults.append(overlapping_boxes_header['text'].to_list() == ['Skattenivå', 'Botnfrådrag'])
        overlapping_boxes_header2 = GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(left_table = overlapping_boxes_header.iloc[0:1,:],right_table = header_area,include_self=True)
        subresults.append(overlapping_boxes_header2['text'].to_list() == ['Skattenivå'] *2)
        overlapping_boxes_header3 = GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(left_table = header_area, right_table = overlapping_boxes_header.iloc[0:1,:],include_self=True)
        subresults.append(overlapping_boxes_header3['text'].to_list() == ['Skattenivå', 'Botnfrådrag'])
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_filter_horizontally_overlapping_on_same_row(self):
        print("Er nå inne i TestGrid.test_filter_horizontally_overlapping_on_same_row")
        header_area = grid_miner_obj_oystre_slidre.extract_header_area()
        #Skal ha fjernet "Skattenivå" og "Botnfrådrag"
        text_filtered_header_area = GridMiner.filter_horizontally_overlapping_on_same_row(header_area)['text'].to_list()
        text_filtered_header_area_comparison = ['Adresse', 'Eigedom', 'Takst', 'Grunnlag', 'Promillesats', 'Skatt', 'Fritak']
        self.assertEqual(text_filtered_header_area,text_filtered_header_area_comparison)     
    #
    def test_apply_median_filter(self):
        print("Er nå inne i TestGrid.test_apply_median_filter")
        filtered_data = mock_grid_miner_obj.apply_regex_filter(mock_grid_miner_obj.tesseract_page_data)
        comparison = mock_grid_miner_obj.tesseract_page_data.copy()
        for regex in mock_grid_miner_obj.required_regex_filter:
            boolean_list = [isinstance(text_string,str) and re.search(pattern =regex,string= text_string) is not None for text_string in comparison['text'].to_list()]
            comparison = comparison.loc[boolean_list]
        #
        subresults = []  
        subresults.append(filtered_data.shape == comparison.shape) 
        subresults.append(list(filtered_data.columns) == list(comparison.columns)) 
        for col in comparison.columns:
            subresults.append((filtered_data[col].to_list() == comparison[col].to_list()))
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_apply_regex_filter(self): 
        print("Er nå inne i TestGrid.test_apply_regex_filter")
        filtered_data = mock_grid_miner_obj.apply_regex_filter(mock_grid_miner_obj.tesseract_page_data)
        row_index_included = []
        enum = 0
        for row in mock_grid_miner_obj.tesseract_page_data.itertuples():
            text = getattr(row,"text")
            all_regex_satisfied = True
            for regex in mock_grid_miner_obj.required_regex_filter:
                if not isinstance(text,str) or re.search(pattern = regex,string = text,flags=re.IGNORECASE) is None:
                    all_regex_satisfied = False
                    break
                #
            if all_regex_satisfied:
                row_index_included.append(enum)
            #
            enum = enum + 1 
        #
        comparison_data = mock_grid_miner_obj.tesseract_page_data.copy().iloc[row_index_included ,:]
        subresults = []
        subresults.append(list(filtered_data.shape) == list(comparison_data.shape))
        for col in comparison_data:
            subresults.append(filtered_data[col].to_list() == comparison_data[col].to_list())
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_filter_boundary_data(self):
        print("Er nå inne i TestGrid.test_filter_boundary_data")
        #Over til filter_boundary_data
        #Tester her ikke med "row_boundaries = False", mend denne funksjonen blir testet separat et annet sted
        filtered_data = grid_miner_obj.filter_boundary_data(pytesseract_data_mock_vefsn.copy(),row_boundaries = True)
        #Testet at ikke overlappende bokser
        query_overlapping_boxes_row = """
            SELECT table1.*
            FROM  filtered_data table1
            INNER JOIN  filtered_data table2 ON
            (
            (table1.right >= table2.left and table1.left <= table2.left) OR
            (table2.right >= table1.left and table2.left <= table1.left)
            )  AND
            (
            (table1.bottom >= table2.top and table1.top <= table2.top) OR
            (table2.bottom >= table1.top and table2.top <= table1.top)
            )
            WHERE
            table1.top <> table2.top OR
            table1.bottom <> table2.bottom OR
            table1.left <> table1.left OR
            table2.right <> table2.right        
            """  
        overlapping_boxes_row =  ps.sqldf(query_overlapping_boxes_row, locals())
        #
        subresults = []  
        subresults.append((overlapping_boxes_row.shape[0] ==0))
        #
        # tester at filtrerer bort verdier slik det skal ffra dataene som er "grunnlag for å sette skillelinjer"
        for row in filtered_data.itertuples():
            text = getattr(row,"text")
            all_regex_satisfied = True
            for regex in mock_grid_miner_obj.required_regex_filter:
                if not isinstance(text,str) or re.search(pattern = regex,string = text,flags=re.IGNORECASE) is None:
                    all_regex_satisfied = False
                break
            #
            subresults.append(all_regex_satisfied)
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_find_row_boundary_candidates(self):
        print("Er nå inne i TestGrid.test_find_row_boundary_candidates")
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
        print("Er nå inne i TestGrid.test_extract_row_boundaries")
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
        print("Er nå inne i TestGrid.test_set_rownum")
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
                break
            #
        #
        subresults.append(equal_values)
        #   
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_extract_non_void_lines(self):
        print("Er nå inne i TestGrid.test_extract_non_void_lines")
        mock_data_no_void_lines = mock_grid_miner_obj.extract_non_void_lines().reset_index(drop=True)
        non_void_lines = []
        row_nr = 0
        for row in mock_pytesseract_data.itertuples():
            if getattr(row,"conf") > 0 and re.search(pattern=r'\S',string = getattr(row,"text")) is not None:
                non_void_lines.append(row_nr)
            #
            row_nr = row_nr + 1
            #
        #
        mock_data_comparison = mock_pytesseract_data.iloc[non_void_lines,:].reset_index(drop=True)
        relevant_columns = [col for col in mock_data_comparison]
        subresults = []
        subresults.append(list(mock_data_no_void_lines.columns)[:len(relevant_columns)] == list(mock_data_comparison.columns))
        for col in relevant_columns:
            subresults.append((mock_data_no_void_lines[col].to_list() == mock_data_comparison[col].to_list()) )
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_create_aggregated_bbox(self):
        print("Er nå inne i TestGrid.test_create_aggregated_bbox")
        #
        aggregated_data = GridMiner.create_aggregated_bbox(multiline_cell_data)
        aggregated_data['text'] = re.sub(pattern = r'\W',repl='',string = aggregated_data['text'])
        subresults = []
        #
        for key in ["left","top","right","bottom","width","height","text"]:
            subresults.append(aggregated_data[key] == comparison_dict[key])
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_create_header_area_with_multiline_column_name(self):
        print("Er nå inne i TestGrid.test_create_header_area_with_multiline_column_name")
        #
        orig_multiline_header_area = grid_miner_obj_multiline.tesseract_page_data.query("conf > 0 and rownum == 5")[['conf','text','top','bottom','left','right','rownum']]
        transformed_multiline_area =  GridMiner.create_header_area_with_multiline_column_names(orig_multiline_header_area)
        multiline_columns = [grid_miner_obj_multiline.clean_column_name(header_value) for header_value in   transformed_multiline_area ['text'].to_list()]
        comparison = ['GnrBnr', 'Adresse', 'Eiendomsskattetakst', 'Obligatoriskreduksjon', 'Skattegrunnlag', 'Beregningsgrunnlag', 'Skattesats', 'Skattebeløp']
        subresults = []
        subresults.append(multiline_columns == comparison)
        #
        gathered_multiline_cell_data  = grid_miner_obj_multiline.create_header_area_with_multiline_column_names(multiline_cell_data)
        gathered_multiline_cell_data['text'] = gathered_multiline_cell_data['text'].map(lambda x: re.sub(pattern = r'\W',repl='',string=x))
        subresults.append(gathered_multiline_cell_data.shape == (1,len(comparison_dict)))
        for col in gathered_multiline_cell_data:
            subresults.append(gathered_multiline_cell_data[col].to_list() == [comparison_dict[col]])
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_gather_multiline_column_names(self):
        print("Er nå inne i TestGrid.test_gather_multiline_column_names")
        header_area_multiline = grid_miner_obj_multiline.extract_header_area().sort_values(by="left")
        multiline_columns = [grid_miner_obj_multiline.clean_column_name(header_value) for header_value in   header_area_multiline['text'].to_list()]
        comparison = ['GnrBnr', 'Adresse', 'Eiendomsskattetakst', 'Obligatoriskreduksjon', 'Skattegrunnlag', 'Beregningsgrunnlag', 'Skattesats', 'Skattebeløp']
        self.assertEqual(multiline_columns,comparison)
    #
    def test_extract_table_area(self):
        print("Er nå inne i TestGrid.test_extract_table_area")
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
    def test_put_header_on_same_row(self):
        print("Er nå inne i TestGrid.test_put_header_in_one_row")
        tesseract_dir_vertical_span = "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/vertical_span_header"
        path_parquet_vertical_span = os.path.join(tesseract_dir_vertical_span,"erroneous_rows.parquet" )
        pytesseract_data_vertical_span = pd.read_parquet(path_parquet_vertical_span,engine= "pyarrow")
        #Memo til self: Metoden "extract_table_area" starter med de original "tesseract-dataene", og har derfor ingen inputargumenter
        grid_miner_obj_vertical_span = GridMiner(tesseract_page_data = pytesseract_data_vertical_span.copy(),skip_lines_top=1,skip_lines_bottom=1,vertical_span_header = 2)
        table_area_vertical_span  = grid_miner_obj_vertical_span.extract_table_area()[['conf','rownum','text','top','left','bottom','right']].query("conf > 0 and rownum == 0").sort_values(by= "left").copy()
        table_area_header =  table_area_vertical_span['text'].to_list()
        comparison_values = ["Adresse","Eigedom","Takst","Skattenivå","Botnfrådrag","Grunnlag","Promillesats","Skatt","Fritak"]
        self.assertEqual(table_area_header,comparison_values)
    #
    def test_reset_rownum(self):
        print("Er nå inne i TestGrid.test_reset_rownum")
        table_area = grid_miner_obj.extract_table_area()
        table_area_new_rownum_values = grid_miner_obj.reset_rownum(table_area)
        sorted_rownum = sorted(table_area_new_rownum_values['rownum'].unique())
        self.assertEqual(sorted_rownum, list(range(len(sorted_rownum))))
    #
    def test_extract_header_area(self):
        print("Er nå inne i TestGrid.test_extract_header_area")
        header_area = grid_miner_obj.extract_header_area()
        header_text = [value.strip() for value in  header_area['text'].to_list()]
        subresults = []
        subresults.append((header_text== column_names))
        #Tester nå også at det går an å "presette" "header area"
        #
        header_area_page2 = grid_miner_obj_page2.extract_header_area()
        #
        subresults.append(header_area_page1.shape == header_area_page2.shape)
        subresults.append(list(header_area_page1.columns) == list(header_area_page2.columns))
        for col in header_area_page1.columns:
            subresults.append(header_area_page1[col].to_list() == header_area_page2[col].to_list())
        #
        print(f"subresults er {subresults}")
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_clean_column_name(self):
        print("Er nå inne i TestGrid.test_clean_column_name")
        column_name =  grid_miner_obj.clean_column_name("~my_column_name.")
        comparison_value = "my_column_name"
        self.assertEqual(column_name, comparison_value)     
    #
    def test_extract_column_boundaries(self):
        print("Er nå inne i TestGrid.test_extract_column_boundaries")
        column_boundaries =  grid_miner_obj.extract_column_boundaries().sort_values(by="left_boundary")
        colnum_values = column_boundaries['colnum'].to_list()
        column_name_values = column_boundaries['column'].to_list()        
        table_area = grid_miner_obj.extract_table_area().query("conf > 0")
        filtered_page_data = grid_miner_obj.filter_boundary_data(table_area)
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
        print("Er nå inne i TestGrid.test_add_colnum")
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
    def test_add_intra_cell_rownum(self):
        print("Er nå inne i TestGrid.test_add_intra_cell_rownum")
        #
        multiline_data_with_colnum = grid_miner_obj_multiline.add_colnum().sort_values(["colnum","rownum"])
        column_multiline_data  = multiline_data_with_colnum.query("column == 'Adresse'")
        column_data_with_intra_cell_rownum = grid_miner_obj_multiline.add_intra_cell_rownum(column_multiline_data)
        # Skal så lage en "sammenligning
        list_column_data_with_intra_cell_rownum = []
        #
        for rownum in sorted(set(column_multiline_data['rownum'])):
            #Plukker ut data med samme "rownum"
            iter_column_multiline_data = column_multiline_data.query(f"conf > 0 and rownum == {rownum}").copy()
            iter_column_multiline_data.columns = Index(['outer_rownum' if col == 'rownum' else col for col in iter_column_multiline_data.columns])
            #Lager "interne radnumre" som deler opp innholdet i gjeldende cell inn i flere linjer hvis noen verdiene i cellen  er vertikalt adskilt fra de andre
            cell_grid_obj = GridMiner(tesseract_page_data = iter_column_multiline_data[['text','conf','left','width','top','height','outer_rownum']].copy(),
                                      apply_row_boundary_filter = False)
            cell_tesseract = cell_grid_obj.tesseract_page_data 
            #Må bytte tilbake til ønskede radnavn
            replacement_dict = {'rownum' : 'intra_cell_rownum', 'outer_rownum' : 'rownum' }
            cell_tesseract.columns = Index([replacement_dict[col] if col in replacement_dict.keys() else col for col in cell_tesseract.columns])
            list_column_data_with_intra_cell_rownum.append(cell_tesseract)
        #
        comparison_data = pd.concat(list_column_data_with_intra_cell_rownum)
        #
        subresults = []
        subresults.append(column_data_with_intra_cell_rownum.shape[0] == comparison_data.shape[0])
        #   
        for col in ['text','conf','left','width','top','height','rownum','intra_cell_rownum']:
            subresults.append(column_data_with_intra_cell_rownum[col].to_list() == comparison_data[col].to_list())
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_extract_column_values(self):
        print("Er nå inne i TestGrid.test_extract_column_values")
        page_table_with_colnum = grid_miner_obj.add_colnum()
        subresults = []
        #Memo til selv: For "sammenblandede kolonner er det beste vi kan håpe at konkatenasjonen "
        jumbled_columns_pairs = [('Grunnlag','Promillesats'),('Skatt','Fritak')]
        jumbled_columns = []
        for jumbled_columns_pair in jumbled_columns_pairs:
            for col in jumbled_columns_pair:
                jumbled_columns.append(col)
            #
        #
        non_jumbled_columns = [col for col in column_names if not col in jumbled_columns]
        subresults = []
        for column in non_jumbled_columns:
            raw_column_values = grid_miner_obj.extract_column_values(page_table_with_colnum,column)
            column_values = [value.strip() for value in raw_column_values] 
            raw_comparison_values =  [value for value in test_values[f"{column}"]]
            #  #Memo til ti                                           
            comparison_values = [str(value).strip() for value in raw_comparison_values]
            comparison_result = (column_values == comparison_values)
            subresults.append(comparison_result)
            if not comparison_result:
                print(f"column er {column},len(column_values) er {len(column_values)} column_values er\n{column_values} og comparison_values er {comparison_values}")
                sys.exit()  
            #
        #
        #Memo til selv: "jumbled_columns" inkludere begge kolonnene med prosenttegn, både "Skattenivå_i_prosent" og "Promillesats"
        for jumbled_columns_pair in jumbled_columns_pairs:
            raw_column_values1 = grid_miner_obj.extract_column_values(page_table_with_colnum,jumbled_columns_pair[0])  
            #Skal nå konkatenere to lister med strenger element for element
            raw_column_values2 = grid_miner_obj.extract_column_values(page_table_with_colnum,jumbled_columns_pair[1])
            #Må både fjerne "etter prosent" og  mellomrom for at det skal fungere
            raw_column_values = [value1 + value2 for value1, value2 in zip(raw_column_values1,raw_column_values2)]
            remove_pattern = r"((%|‰).*)|\s+"
            #
            column_values = [re.sub(pattern=remove_pattern,repl='',string= value) for value in raw_column_values]
            raw_comparison_values1 =  [value for value in test_values[f"{jumbled_columns_pair[0]}"]]
            raw_comparison_values2 =  [value for value in test_values[f"{jumbled_columns_pair[1]}"]]
            raw_comparison_values = [value1 + value2 for value1, value2 in zip(raw_comparison_values1,raw_comparison_values2)]
            comparison_values = [re.sub(pattern=remove_pattern,repl='',string= value) for value in raw_comparison_values]
            #  må 
            #Strengeverdier som inneholder spesialtegnet '§' kan ikke forventes å bli lest riktig inn
            #Memo til slutt: OCR sliter veldig med §
            filtered_comparison_tuples = [(column_value,comparison_value)  for column_value,comparison_value in zip(column_values,comparison_values) if re.search(pattern=r'§',string= comparison_value) is None]
            comparison_results =   [tupple_value[0] == tupple_value[1] for tupple_value in  filtered_comparison_tuples  ]
            subresults.append(comparison_results)
            if not pd.Series(comparison_results).all():
                print(f"Sammenblandingskolonne1 er {jumbled_columns_pair[0]},og sammenblandingskolonne2 er {jumbled_columns_pair[1]}\nlen(column_values) er {len(column_values)} column_values er\n{column_values} og comparison_values er {comparison_values}")
                sys.exit()  
            #
        #
        print(f"Før evaluering av subresults så er alle subresults i test_extract_column_values\n{subresults}")
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_gather_multiline_cells(self):
        print("Er nå inne i TestGrid.test_gather_multiline_cells")
        page_table_with_colnum = grid_miner_obj_multiline_cell.add_colnum().sort_values(["colnum","rownum"])
        page_table_with_colnum['text'] = page_table_with_colnum['text'].map(lambda x: re.sub(pattern = r'\W',repl = '',string = x))
        comparison = page_table_with_colnum.copy()
        page_table_with_multiline_cells = grid_miner_obj_multiline_cell.gather_multiline_cells(page_table_with_colnum)
        done_grouping = False
        while not done_grouping:
            unique_rownums_larger_than_one = [rownum for rownum in comparison['rownum'].unique() if rownum > 1]
            done_grouping = True
            for rownum in  unique_rownums_larger_than_one:
                count_unique_colnums = len(comparison.query(f"rownum == {rownum}")['colnum'].unique())
                if count_unique_colnums <= grid_miner_obj_multiline_cell.row_grouping_criteria:
                    comparison['rownum'] = comparison['rownum'].map(lambda x: x-1  if x == rownum  else  x)
                    done_grouping = False          
                    break
                #
            #
        #
        subresults = []
        subresults.append(page_table_with_multiline_cells.shape == comparison.shape)
        for col in page_table_with_multiline_cells.columns:
            subresults.append(page_table_with_multiline_cells[col].to_list() == comparison[col].to_list())
        #
        self.assertTrue(pd.Series(subresults).all())
        #warnings.warn("Ingen test for GridMiner.gather_multiline_cellser implementert!", UserWarning)
    #
    def test_extract_table(self):
        print("Er nå inne i TestGrid.test_extract_table")
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
