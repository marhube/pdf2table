import os
import sys
import re
#
import pandas as pd
import pyarrow 
import numpy as np
import warnings
#**********

#Forsøk på  systematisk testing
from pdf2table.municipal_appraisals import Standard_pdf
#
import unittest
#

test_data_dir = "/home/m01315/General_Python/Packages/pdf2table/test_data"
path_initial_table = os.path.join(test_data_dir,'nordre_land_intial_table.csv')
path_header = os.path.join(test_data_dir,"nordre_land_columns.txt")
#
knr = '3448'
kommunenavn = "NORDRE LAND"
publication_year = 2024
#
Standard_pdf_obj = Standard_pdf(path_initial_table=path_initial_table,path_header=path_header,knr=knr,kommunenavn = kommunenavn,publication_year=publication_year)
#
Standard_pdf_obj_melhus = Standard_pdf(
    path_initial_table = os.path.join(test_data_dir,'melhus_intial_table.csv'),
    path_header = os.path.join(test_data_dir,"melhus_columns.txt"),
    knr = '5028',
    kommunenavn = "MELHUS",
    publication_year = 2024,
    missing_columns = ['Skattenivå_i_prosent']
)
#
Standard_pdf_obj_oystre_slidre = Standard_pdf(
    path_initial_table= "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/vertical_span_header/øystre_slidre_intial_table.csv",
    path_header= "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/vertical_span_header/øystre_slidre_columns.txt",
    knr='3453',
    kommunenavn="ØYSTRE SLIDRE",
    publication_year=2020,
    replacement_dict = {'Eigedom':'GID','Skattenivå': 'Skattenivå_i_prosent','Botnfrådrag': 'Bunnfradrag'}
)
#
Standard_pdf_obj_holmestrand = Standard_pdf(
    path_initial_table= "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/municipal_appraisals_holmestrand/holmestrand_intial_table.csv",
    path_header= "/home/m01315/General_Python/Packages/pdf2table/explore_output/Scripts/municipal_appraisals_holmestrand/holmestrand_columns.txt",
    knr='3903',
    kommunenavn="HOLMESTRAND",
    publication_year=2023,
    replacement_dict = {'Gnr':'gnr',"Bnr": "bnr","Fnr": "festenr" ,"Snr": "snr",'Skattenivå': 'Skattenivå_i_prosent','Skattegrunnlag': "Grunnlag"}
)
#
class TestMunicipalAppraisals(unittest.TestCase):
    #
    def test_instantiation(self):
        print("Er nå inne i TestMunicipalAppraisals.test_instantiation")
        self.assertIsInstance(Standard_pdf_obj,Standard_pdf)
    #
    def test_clean_column_name(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_clean_column_name")
        cleaned_column_name = Standard_pdf_obj.clean_column_name("~Promillesats ")
        comparison = "Promillesats"
        self.assertEqual(cleaned_column_name,comparison)
    #
    def test_set_header(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_set_header")
        Standard_pdf_obj.set_header()
        column_names = list(Standard_pdf_obj.header)
        comparison = ['Adresse', 'Eiendom', 'Takst', 'Skattenivå', 'Bunnfradrag', 'Grunnlag', 'Promillesats', 'Skatt', 'Fritak']
        self.assertEqual(column_names,comparison)
    #Sjekker at "initial_table" er  en "dataramme" med riktig antall kolonner
    def test_set_initial_table(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_set_initial_table")
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        initial_table  =  pd.read_csv(Standard_pdf_obj.path_initial_table,sep=";",header=None,engine="pyarrow")
        #
        subresults = [] 
        subresults.append(isinstance(initial_table,pd.DataFrame))
        subresults.append((initial_table.shape[0] > 0))
        subresults.append((initial_table.shape[1] == len(Standard_pdf_obj.header)))
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_replace_colnames(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_replace_colnames")
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.replace_colnames()
        comparison = ['Adresse', 'GID', 'Takst', 'Skattenivå_i_prosent', 'Bunnfradrag', 'Grunnlag', 'Promillesats', 'Skatt', 'Fritak']                
        self.assertEqual(list(Standard_pdf_obj.initial_table.columns),comparison)
    #
    def test_resolve_missing_column_dependencies(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_resolve_missing_column_dependencies")
        #
        Standard_pdf_obj_melhus.resolve_missing_column_dependencies()
        #
        percent_columns = Standard_pdf_obj_melhus.percent_columns
        percent_columns_comparison = ['Bunnfradrag', 'Grunnlag', 'Promillesats']
        jumbled_column_pairs  =  Standard_pdf_obj_melhus.jumbled_column_pairs
        jumbled_column_pairs_comparison = [('Grunnlag', 'Promillesats'), ('Skatt', 'Fritak')]
        subresults = [] 
        subresults.append((percent_columns == percent_columns_comparison))
        subresults.append((jumbled_column_pairs == jumbled_column_pairs_comparison))
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_create_missing_columns(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_create_missing_columns")
        Standard_pdf_obj_melhus.set_header()
        Standard_pdf_obj_melhus.set_initial_table()
        Standard_pdf_obj_melhus.create_missing_columns()
        melhus_columns = list(Standard_pdf_obj_melhus.initial_table.columns)
        melhus_columns_comparison = ['Adresse','Eiendom','Takst','Bunnfradrag','Grunnlag','Promillesats','Skatt','Fritak','Skattenivå_i_prosent']
        self.assertEqual(melhus_columns,melhus_columns_comparison)
    #
    def test_clean_percent_string(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_clean_percent_string")
        str_values = ['3%o','3%0','3.4 %o','4 % u','5']
        test_values = []
        for str_value in str_values:
            test_values.append(Standard_pdf.clean_percent_string(str_value))
        #
        comparison_values = ['3','3','3.4','4','5']
        self.assertEqual(test_values,comparison_values)
    #
    def test_clean_percent_columns(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_clean_percent_columns")
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.replace_colnames()
        Standard_pdf_obj.clean_percent_columns()
        #
        percent_columns = ['Skattenivå_i_prosent','Promillesats']
        sample_size = 5
        comparison_values = {
            percent_columns[0] : ['70']*sample_size,
            percent_columns[1]  : ["3,5"]*sample_size
        }
        #        
        subresults = []
        for col in percent_columns:
            cleaned_values = Standard_pdf_obj.initial_table[col].to_list()[:sample_size]
            subresults.append((cleaned_values == comparison_values[col]))
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_preclean_numeric_string(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_preclean_numeric_string")
        test_values = ["v"," v"," v ","v 34","0o"," ö"]
        comparison = ["0"]*3 + ["0 34","0o","0"]
        cleaned_values = []
        for test_value in test_values:
            cleaned_values.append(Standard_pdf.preclean_numeric_string(test_value))
        #
        self.assertEqual(cleaned_values,comparison)
    #
    def test_digitize_string(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_digitize_string")
        test_values = ["kr 1 322 400"]
        comparison = ["1322400"]
        cleaned_values = []
        for test_value in test_values:
            cleaned_values.append(Standard_pdf_obj.digitize_string(test_value))
        #
        self.assertEqual(cleaned_values,comparison)
    #
    def test_extract_initial_zero(self):
        print(f"Er nå inne i TestMunicipalAppraisals.test_extract_initial_zero")
        string_with_initial_zero1 = "08 5 h - Gårds-/Skogbruk"
        string_with_initial_zero2 = "0 Ingen"
        tax_value1,exemption_value1 = Standard_pdf_obj.extract_initial_zero(string_with_initial_zero1)
        tax_value2,exemption_value2 = Standard_pdf_obj.extract_initial_zero(string_with_initial_zero2)
        comparison1 = ('0','8 5 h - Gårds-/Skogbruk')
        comparison2 = ('0','Ingen')
        subresults = [] 
        subresults.append((tax_value1,exemption_value1) == comparison1)
        subresults.append((tax_value2,exemption_value2) == comparison2)
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_looks_like_positive_integer(self):
        print("Er nå inne i TestMunicipalAppraisals.test_looks_like_positive_integer")
        test_values = ['854','08','Ingen','2']
        test_result1 =  Standard_pdf_obj.looks_like_positive_integer(test_values[0],0)
        subresults = [] 
        subresults.append(test_result1) #test_result1 skal være True
        test_result2 = not Standard_pdf_obj.looks_like_positive_integer(test_values[1],0)
        subresults.append(test_result2) 
        test_result3 =  not Standard_pdf_obj.looks_like_positive_integer(test_values[2],0)
        subresults.append(test_result3) 
        test_result4 = Standard_pdf_obj.looks_like_positive_integer(test_values[3],0)
        subresults.append(test_result4) 
        test_result5 = not Standard_pdf_obj.looks_like_positive_integer(test_values[3],1)
        subresults.append(test_result5)
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_looks_like_float(self):
        print("Er nå inne i TestMunicipalAppraisals.test_looks_like_float")
        string_values =  ['3,5','3.5','3,5,2','2.2c']
        test_values =  [Standard_pdf.looks_like_float(string_value) for string_value in string_values]
        comparison_values = [True,True,False,False]
        self.assertTrue(test_values == comparison_values )
    #
 
    def test_extract_from_jumbled_value(self):
        print("Er nå inne i TestMunicipalAppraisals.test_extract_from_jumbled_value")
        test_value1 = "2 854 Ingen"
        test_value2 = '08 5 h - Gårds-/Skogbruk'
        test_value3 = '0 Ingen'
        subresults = [] 
        tax_value1,exemption_value1 =  Standard_pdf_obj.extract_from_jumbled_value(test_value1)
        comparison1 = ("2 854","Ingen")
        subresults.append(((tax_value1,exemption_value1) ==  comparison1)) 
        tax_value2,exemption_value2 =  Standard_pdf_obj.extract_from_jumbled_value(test_value2)
        comparison2 = ("0","8 5 h - Gårds-/Skogbruk")
        subresults.append(((tax_value2,exemption_value2) ==  comparison2)) 
        tax_value3,exemption_value3 =  Standard_pdf_obj.extract_from_jumbled_value(test_value3)
        comparison3 = ("0","Ingen")
        subresults.append(((tax_value3,exemption_value3) ==  comparison3)) 
        #
        self.assertTrue(pd.Series(subresults).all()) 
    #
    def test_clean_jumbled_column_pairs(self):
        print("Er nå inne i TestMunicipalAppraisals.test_clean_jumbled_column_pairs")
        comparison_tax_values = ["2 854","0","0"]
        comparison_exemption_values = ["Ingen","8 5 h - Gårds-/Skogbruk","Ingen"]
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.replace_colnames()
        Standard_pdf_obj.clean_percent_columns()
        Standard_pdf_obj.set_cadastre_values()
        Standard_pdf_obj.set_tax_year()
        Standard_pdf_obj.preclean_numeric_columns()
        Standard_pdf_obj.clean_jumbled_column_pairs()
        #
        top_tax_exemption_values = Standard_pdf_obj.initial_table[['Skatt','Fritak']].head(len(comparison_tax_values))
        subresults = [] 
        subresults.append(top_tax_exemption_values['Skatt'].to_list() == comparison_tax_values )
        subresults.append(top_tax_exemption_values['Fritak'].to_list() == comparison_exemption_values )   
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_extract_GID_part(self):
        print("Er nå inne i TestMunicipalAppraisals.test_extract_GID_parts")
        nvalues = 10
        comparison_values = {
            'gnr':[1]*nvalues,
            'bnr': [1,2,3,4,5,5,6,7,8,12],
            'festenr' : [0]*nvalues,
            'snr' : [0]*nvalues
              }
        #
        comparison_values['festenr'][5] = 1
        # Må gjøre om til strengeverdier
        for key in comparison_values.keys():
            comparison_values[key] = [str(value) for value in comparison_values[key]]
        #
        gnr = Standard_pdf_obj.initial_table['GID'].map(lambda x: Standard_pdf_obj.extract_GID_part(x,0,2)).to_list()[:nvalues]
        bnr = Standard_pdf_obj.initial_table['GID'].map(lambda x:   Standard_pdf_obj.extract_GID_part(x,1,2)).to_list()[:nvalues]
        festenr = Standard_pdf_obj.initial_table['GID'].map(lambda x:  Standard_pdf_obj.extract_GID_part(x,2,4)).to_list()[:nvalues]
        snr = Standard_pdf_obj.initial_table['GID'].map(lambda x:  Standard_pdf_obj.extract_GID_part(x,3,4)).to_list()[:nvalues]
        extracted_values_dict = {'gnr':gnr,'bnr': bnr,'festenr': festenr,'snr':snr}
        subresults = [] 
        for col in ['gnr','bnr','festenr','snr']:
            subresults.append(extracted_values_dict[col] == comparison_values[col] )
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_set_cadastre_values(self):
        print("Er nå inne i TestMunicipalAppraisals.test_set_cadastre_values")
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.replace_colnames()
        Standard_pdf_obj.clean_percent_columns()
        Standard_pdf_obj.set_cadastre_values()
        nvalues = 10
        comparison_values = {
            'gnr':[1]*nvalues,
            'bnr': [1,2,3,4,5,5,6,7,8,12],
            'festenr' : [0]*nvalues,
            'snr' : [0]*nvalues
              }
        #
        comparison_values['festenr'][5] = 1
        # Må gjøre om til strengeverdier
        for key in comparison_values.keys():
            comparison_values[key] = [str(value) for value in comparison_values[key]]
        #
        subresults = [] 
        for col in ['gnr','bnr','festenr','snr']:
            subresults.append(Standard_pdf_obj.initial_table[col].to_list()[:nvalues] == comparison_values[col] )
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_set_GID_values(self):
        print("Er nå inne i TestMunicipalAppraisals.test_set_GID_values")
        Standard_pdf_obj_holmestrand.set_header()
        Standard_pdf_obj_holmestrand.set_initial_table()
        Standard_pdf_obj_holmestrand.replace_colnames()
        comparison_table = Standard_pdf_obj_holmestrand.initial_table.copy()
        GID_values = []
        for row in comparison_table.itertuples():
            cadastre_values = []
            for col in ['gnr','bnr','festenr','snr']:
                cadastre_values.append(re.sub(pattern=r'\D',repl='',string=getattr(row,col)))
            #
            GID_values.append(r'/'.join(cadastre_values))
        #
        comparison_table['GID'] = pd.Series(GID_values,dtype = 'str')
        #        
        Standard_pdf_obj_holmestrand.set_GID_values()
        self.assertEqual(Standard_pdf_obj_holmestrand.initial_table['GID'].to_list(),comparison_table['GID'].to_list())
    #
    def test_set_tax_year(self):
        print("Er nå inne i TestMunicipalAppraisals.test_set_tax_year")
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.set_tax_year()
        comparison_values = [Standard_pdf_obj.publication_year] * Standard_pdf_obj.initial_table.shape[0]
        self.assertEqual(Standard_pdf_obj.initial_table['Skatteår'].to_list() ,comparison_values)
    #
    def test_set_municip_values(self):
        print("Er nå inne i TestMunicipalAppraisals.test_set_municip_values")
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.set_municip_values()
        comparison_dict = {
            'knr' : [Standard_pdf_obj.knr] * Standard_pdf_obj.initial_table.shape[0],
            'kommunenavn' : [Standard_pdf_obj.kommunenavn] * Standard_pdf_obj.initial_table.shape[0]           
        }
        subresults = []
        for col in ['knr','kommunenavn']:
            subresults.append(Standard_pdf_obj.initial_table[col].to_list() ==  comparison_dict[col])
        #
        self.assertTrue(pd.Series(subresults).all())
    #Memo til selv: "clean_and_convert_float","clean_and_convert_integer"  er statiske metoder
    def test_clean_and_convert_float(self):
        print("Er nå inne i TestMunicipalAppraisals.test_clean_and_convert_float")
        string_values =  ['3,5','3.5','3,5,2','2']
        test_values =  [Standard_pdf.clean_and_convert_float(string_value) for string_value in string_values]
        comparison_values = [3.5] * 2 + [None,2.0] 
        self.assertEqual(test_values,comparison_values)
    #
    def test_clean_and_convert_integer(self):
        print("Er nå inne i TestMunicipalAppraisals.test_clean_and_convert_integer")
        string_values =  ['3','3578','3578689','4412877,00']
        test_values =  [Standard_pdf.clean_and_convert_integer(string_value) for string_value in string_values]
        comparison_values = [3,3578,3578689,4412877]
        self.assertEqual(test_values,comparison_values)
    #
    def test_clean_and_convert2num(self):
        print("Er nå inne i TestMunicipalAppraisals.test_clean_and_convert2num")
        string_values = ['3,5','3.5','3,5,2'] + ['3','3 578','3 578 689'] + [" ","0v","" ]
        test_values =  [Standard_pdf.clean_and_convert2num(string_value) for string_value in string_values]
        comparison_values = [3.5] * 2 + [None] + [3,3578,3578689] + [None]*3
        self.assertEqual(test_values,comparison_values)
    #
    def test_conditionally_clean_and_convert2num(self):
        print("Er nå inne i TestMunicipalAppraisals.test_conditionally_clean_and_convert2num")
        #Må først rense dataene litt
        Standard_pdf_obj.set_header()
        Standard_pdf_obj.set_initial_table()
        Standard_pdf_obj.replace_colnames()
        Standard_pdf_obj.clean_percent_columns()
        Standard_pdf_obj.set_cadastre_values() 
        Standard_pdf_obj.set_tax_year()   #Må også legge til skatteår
        initial_table =   Standard_pdf_obj.initial_table.copy()        
        #Tester først med strengeverdier
        subresults = []
        string_values = ['3,5','3.5','3,5,2'] + ['3','3 578','3 578 689'] + [" ","0v","" ]
        list_integer_col = [False]*3 + [True]*3 + [True]*3
        test_values =  [Standard_pdf.clean_and_convert2num(string_value,integer_col) for string_value,integer_col in zip(string_values,list_integer_col)]
        comparison_values = [3.5] * 2 + [None] + [3,3578,3578689] + [None]*3
        subresults.append(test_values == comparison_values)
        #Skal egentlig her først og fremst teste at ikke tester
        for num_col in Standard_pdf_obj.numeric_columns:
            print(f"num_col er {num_col}")
            if num_col in Standard_pdf_obj.integer_columns:
                list_int_values = [Standard_pdf.conditionally_clean_and_convert2num(string_val,integer_col=True)  for string_val in initial_table[num_col].to_list()]
                initial_table[num_col] = pd.array(list_int_values, dtype=pd.Int64Dtype())
            else:
                initial_table[num_col] = initial_table[num_col].map(lambda x: Standard_pdf.conditionally_clean_and_convert2num(x,integer_col=False))
        #
        initial_table2 = initial_table.copy()
        for num_col in Standard_pdf_obj.numeric_columns:
            print(f"num_col er {num_col}")
            if num_col in Standard_pdf_obj.integer_columns:
                list_int_values = [Standard_pdf.conditionally_clean_and_convert2num(string_val) for string_val in initial_table2[num_col].to_list()]
                initial_table2[num_col] = pd.array(list_int_values, dtype=pd.Int64Dtype())
            else:
                initial_table2[num_col] = initial_table2[num_col].map(Standard_pdf.conditionally_clean_and_convert2num)
            #
        #
        subresults.append(list(initial_table2.shape) == list(initial_table.shape))
        subresults.append(list(initial_table2.columns) == list(initial_table.columns))
        for col in initial_table.columns:
            #Må fjerne "nan"-verdier for å kunne sammenligne
            boolean_values = [(pd.isna(value1) and pd.isna(value2)) or value1 == value2 for value1,value2 in zip(initial_table[col].to_list(),initial_table2[col].to_list())  ]
            subresults.append(boolean_values)
            if not pd.Series(boolean_values).all():
                print(f"col er {col}")
                print(f"initial_table[col].to_list() er {initial_table[col].to_list()}\og")
                print(f"initial_table2[col].to_list() er {initial_table2[col].to_list()}\og")
                sys.exit()     
        #
        self.assertTrue(pd.Series(subresults).all())
    #
    def test_create_standard_table(self):
        print("Er nå inne i TestMunicipalAppraisals.test_create_standard_table")
        Standard_pdf_obj.create_standard_table()
        standard_table = Standard_pdf_obj.initial_table.copy()
        subresults = []
        all_included_and_correct_type = True
        #
        for col in Standard_pdf_obj.relevant_columns:
            if not col in standard_table.columns or not (
                (col in Standard_pdf_obj.integer_columns and re.search(pattern= r'^int(\d{2})?$',string= str(standard_table[col].dtype),flags=re.IGNORECASE)  is not None) or  #Godtar nå int, int32,in64 'Int64' or 
                (col in Standard_pdf_obj.numeric_columns and not col in Standard_pdf_obj.integer_columns and re.search(pattern= r'^float(\d{2})?$',string= str(standard_table[col].dtype),flags=re.IGNORECASE)  is not None)   or 
                (col not in Standard_pdf_obj.numeric_columns and str(standard_table[col].dtype) == 'object')
                ):
                    all_included_and_correct_type = False
                    print(f"Kolonnen {col} er enten ikke med eller har ikke riktig datatype")
                    break
                #
            #
        #
        subresults.append(all_included_and_correct_type)
        self.assertTrue(pd.Series(subresults).all())
    #
#


        
              

        

#
    


