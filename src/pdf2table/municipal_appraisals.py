import re
import sys
import warnings
#
import pandas as pd
from pandas import Index # For å gjøre mypy fornøyd
import pyarrow  
import numpy as np
#
from typing import Dict,Self,Tuple
#
from pdf2table.grid import CustomizedException
#
def raise_missing_columns_in_standard_table_exception(missing_columns: list[str]) -> None:
    raise CustomizedException(f"The following columns are missing from the standard table{', '.join(missing_columns)}.")
#
def raise_wrong_datatypes_standard_table_exception(columns_wrong_datatype: list[str]) -> None:
    raise CustomizedException(f"The following columns have the wrong datatype {', '.join(columns_wrong_datatype)}.")
#
# Memo til selv: Tar nå  i "prosentrensen" av dataene også hensyn til at "prosentkolonner kan bli blandet sammen med nabokolonnen"
class Standard_pdf:
    def __init__(
        self,
        path_initial_table :str,
        path_header: str,
        knr : str,
        kommunenavn: str,
        publication_year: int,
        replacement_dict : Dict[str,str] = {'Eiendom':'GID','Skattenivå': 'Skattenivå_i_prosent'},
        float_columns: str|list[str] = ['Skattenivå_i_prosent','Promillesats'],
        integer_columns : str|list[str] = ['gnr','bnr','festenr','snr','Takst','Skattenivå_i_prosent','Bunnfradrag','Grunnlag','Skatt','Skatteår'],
        jumbled_column_pairs  : tuple[str,str]|list[tuple[str,str]] = [('Skattenivå_i_prosent','Bunnfradrag'),('Grunnlag','Promillesats'),('Skatt','Fritak')], #Kolonnepar der den ene er klint opp i den andre og derfor kan være vanskelig å skille
        percent_columns :  str|list[str] = ['Skattenivå_i_prosent','Bunnfradrag','Grunnlag','Promillesats'], 
        relevant_columns : str|list[str] = ['knr','kommunenavn','GID','gnr','bnr','festenr','snr','Takst','Skattenivå_i_prosent','Bunnfradrag','Grunnlag','Promillesats','Skatt','Skatteår'],
        missing_columns : str|list[str] = [],
        clean_header_regex : str|list[str] = r'\W', #Rens vekk evt spesialtegn fra kolonnenavn
        GID_separator: str = "/",
        cadastre_values_already_set : bool = False,
        output_GID_separator: str = "/"    ,
        remove_missing_column_from_jumbled_pair : bool = True # Veldig teknisk parameter
        ) :
        #Memo til selv: Gjør først str|list om til list
        if isinstance(float_columns,str):
            float_columns = [float_columns]
        #
        if isinstance(integer_columns,str):
            integer_columns = [integer_columns]
        #
        if isinstance(percent_columns,str):
            percent_columns = [percent_columns]
        #
        if isinstance(relevant_columns,str):
            relevant_columns = [relevant_columns]
        #
        if isinstance(missing_columns,str):
            missing_columns = [missing_columns]
        #
        if isinstance(clean_header_regex,str):
            clean_header_regex = [clean_header_regex]
        #
        if isinstance(jumbled_column_pairs,tuple):
            jumbled_column_pairs = [jumbled_column_pairs]
        #
        self.path_initial_table = path_initial_table
        self.path_header = path_header
        self.knr = knr
        self.publication_year = publication_year
        self.kommunenavn = kommunenavn
        self.replacement_dict = replacement_dict
        self.integer_columns = integer_columns
        self.numeric_columns = integer_columns + float_columns
        self.percent_columns : list  = percent_columns
        #
        self.jumbled_column_pairs  = jumbled_column_pairs 
        self.relevant_columns = relevant_columns
        self.missing_columns = missing_columns
        self.clean_header_regex = clean_header_regex
        self.GID_separator = GID_separator
        self.cadastre_values_already_set = cadastre_values_already_set
        self.output_GID_separator = output_GID_separator
        self.remove_missing_column_from_jumbled_pair = remove_missing_column_from_jumbled_pair
        #
    #OBSS bør ha sin egen enhetstest
    def clean_column_name(self,column_name: str) -> str:
        #
        #Renser kolonnenavnene for spesialtegn (dette kan bortvelges i init)
        for regex in self.clean_header_regex:
            column_name = re.sub(pattern=regex,repl='',string=column_name)
        #
        return column_name
    #
    def set_header(self) -> Self:
        # Open the file in read mode
         with open(self.path_header, "r") as file:
            # Read each line and strip the newline character
            header_line = [text_line.strip() for text_line in file if len(text_line.strip()) > 0]
            #Renser kolonnenavnene for spesialtegn (dette kan bortvelges i init)
            column_name_strings = header_line[0].split(";")
            for regex in self.clean_header_regex:
                column_name_strings = [re.sub(pattern=regex,repl='',string=column_name_string)  for column_name_string in  column_name_strings]
            #Gjør om til "Index" for at "mypy" skal bli fornøyd
            column_names: Index[str] =  Index([self.clean_column_name(column_name) for column_name in header_line[0].split(";")])
            self.header = column_names
            return self
         #
    #
    def set_initial_table(self) -> Self: 
        #Memo til selv: Setter  keep_default_na=False slik at tomme strenger ("") skal forbli tomme strenger under innlesning      
        initial_table  =  pd.read_csv(self.path_initial_table,sep=";",header=None,dtype=str,keep_default_na=False,engine="pyarrow")
        initial_table.columns = self.header
        # Replace None or NaN values explicitly with empty strings
        self.initial_table = initial_table
        return self
    #
    def replace_colnames(self) -> Self:
        new_columns = []
        for col in self.initial_table.columns:
            new_col = col
            if col in self.replacement_dict.keys():
                new_col = self.replacement_dict[col]
                #
            new_columns.append(new_col) 
        #        
        self.initial_table.columns =  Index(new_columns) #For å unngå at mypy skal klage
        return self
    #Memo til selv: "resolve_missing_column_dependencies" rydder opp i "rusk" forsårsaket at det er noen kolonner
    # som ikke finnes i pdf-en
    def resolve_missing_column_dependencies(self) -> Self:
        for missing_column in self.missing_columns:
            if  missing_column in self.percent_columns:
                self.percent_columns.remove(missing_column)
            #
            if self.remove_missing_column_from_jumbled_pair:
                for jumbled_column_pair in self.jumbled_column_pairs:
                    if missing_column in jumbled_column_pair:
                        self.jumbled_column_pairs.remove(jumbled_column_pair)
                    #
                #
            #
        #
        return self
    #
    #Memo til selv: Må i create_missing_columns også "rydde opp" konsekve
    def create_missing_columns(self) -> Self:
        for column in self.missing_columns:
            self.initial_table[column] = ""
        #
        return self
    #
    #Memo til selv: Her følger en rekke kjempeesoteriske funksjoner
    def clean_percent_columns(self) -> Self:
        for column in self.percent_columns:
            self.initial_table[column] = self.initial_table[column].map(lambda x: re.sub(pattern=r'(%|‰).*',repl='',string=x))
        #        
        return self
    #
    @staticmethod  #Bruk på alle numeriske kolonner
    def digitize_string(raw_string: str) -> str:
        #Bytter ut "isolert v" med "0".
        digitized_string = raw_string.strip()
        #Tar bort forstavelse "kr" og anndre
        remove_patterns = [r'^kr\s*']
        for remove_pattern in remove_patterns:
            digitized_string = re.sub(pattern=remove_pattern,repl="",string=digitized_string,flags=re.IGNORECASE)
        #
        # #Bytter ut "isolert enkeltbokstav" med "0".
        zero_patterns = [r'^(o|ö)[a-z]',r"^([a-z]|ö)$",r"(?<=\s)([a-z]|ö)$",r"^([a-z]|ö)(?=\s)" ,r"(?<=\s)([a-z]|ö)(?=\s)"]
        for zero_pattern in zero_patterns:
            digitized_string = re.sub(pattern=zero_pattern,repl="0",string=digitized_string,flags=re.IGNORECASE)
        #
        return digitized_string            
    #
    def digitize_numeric_columns(self) -> Self:
        for column in self.numeric_columns:
            #Må sjekke at ikke allerede er numerisk
            if str(self.initial_table[column].dtype) == 'object':
                self.initial_table[column] = self.initial_table[column].map(lambda x: Standard_pdf.digitize_string(x))
            #
        #
        return self
    #
    @staticmethod
    def extract_initial_zero(combined_string : str) -> Tuple[str,str]:
        zero_value ='0'
        remaining_string = re.sub(pattern="^0\s*",repl='',string=combined_string)
        return zero_value,remaining_string   
    #
    @staticmethod
    #Memo tl selv: Forutsetter her at allerede har håndtert tilfellet der "skattestrengen" begynner med "0".
    def looks_like_positive_integer(substr: str,internal_index: int) -> bool:
        passes_test = False
        if len(substr) > 0 and re.search(pattern=r'\D+',string = substr) is None: # ser ut som en sifferrekke
            # Tallet kan ikke starte med 0. Beløp større enn 1 million må være i et helt antall tusen kroner
            if (internal_index== 0 and (re.search(pattern=r'^[1-9]',string= substr) is not None)) or (
                internal_index >= 1 and re.search(pattern=r'^\d{3}$',string=substr) is not None):
                passes_test = True
            #
        #
        return passes_test
    #
    @staticmethod
    def looks_like_float(substr: str) -> bool:
        float_value = Standard_pdf.clean_and_convert2num(substr,integer_col= False)
        return isinstance(float_value,float)
    #
    @staticmethod
    def extract_from_jumbled_value(combined_string) -> Tuple[str,str]:
        left_value = ''
        right_value = ''
        if combined_string.startswith("0"): #Spesialtilfelle. Isåfall
            left_value,right_value = Standard_pdf.extract_initial_zero(combined_string)
        else:
            #Memo til selv: Hvis combined_string ikke begynner med "0" så skal hver "space_splist" enten være fullstendig med i "tax_value" eller helt med i "right_value"
            left_value_splits = []
            space_splits = re.split(pattern=r'\s+',string = combined_string)
            remaining_space_splits = space_splits.copy()
            for enum,space_split in enumerate(space_splits):                
                if Standard_pdf.looks_like_positive_integer(space_split,enum) or Standard_pdf.looks_like_float(space_split):
                    left_value_splits.append(space_split)
                    #Fjerner første element i "remaining_space_splits"
                    remaining_space_splits.pop(0)
                #Memo til selv: I alle andre tilfelelr enn hvis verdien er en "sifferrekke" (positivt heltall) så skal det ikke hentes inne mer enn én "space_split"
                if not Standard_pdf.looks_like_positive_integer(space_split,enum):
                    break                    
               #
            #
            left_value = ' '.join(left_value_splits)
            right_value = ' '.join(remaining_space_splits)             
        #
        return left_value,right_value 
    #
    def clean_jumbled_column_pairs(self) -> Self:   
        for jumbled_column_pair in self.jumbled_column_pairs:
            left_column = jumbled_column_pair[0]
            right_column = jumbled_column_pair[1]
            list_left_column_values = []
            list_right_column_values = []   
            for row in self.initial_table.itertuples():
                left_column_value = getattr(row,left_column)
                right_column_value = getattr(row,right_column)
                if left_column_value  == "" or right_column_value  == "":
                    left_column_value,right_column_value = Standard_pdf.extract_from_jumbled_value(left_column_value  +  right_column_value)
                #
                list_left_column_values.append(left_column_value)
                list_right_column_values.append(right_column_value)
            #     
            self.initial_table[left_column] = list_left_column_values
            self.initial_table[right_column] = list_right_column_values            
        #
        return self
    #Over til kolonner relatert til matrikkeladresse
    #
    def extract_GID_part(self,GID_string: str,component_index : int,required_split_count : int = 0) -> str:
        #Sjekker om har nok forekomster av "/"
        required_length = max(required_split_count,component_index+1)
        GID_component_value = ''
        GID_splits = GID_string.split(self.GID_separator)
        if len(GID_splits) >= required_length:
            GID_component_value = GID_splits[component_index]
        #
        return GID_component_value
    #
    def set_cadastre_values(self) -> Self:
        self.initial_table['gnr'] = self.initial_table['GID'].map(lambda x: self.extract_GID_part(x,0,2))
        self.initial_table['bnr'] = self.initial_table['GID'].map(lambda x:   self.extract_GID_part(x,1,2))
        self.initial_table['festenr'] = self.initial_table['GID'].map(lambda x:  self.extract_GID_part(x,2,4))
        self.initial_table['snr'] = self.initial_table['GID'].map(lambda x:  self.extract_GID_part(x,3,4))
        #Endre nå koden til også å gi en "standard utforming" på "GID-separator"
        if self.GID_separator  != self.output_GID_separator:
            self.initial_table['GID'] = self.initial_table['GID'].map(lambda x: x.replace(self.GID_separator,self.output_GID_separator))

        #        
        return self
    # 
    def set_GID_values(self) -> Self:
        GID_values = []
        for row in self.initial_table.itertuples():
            cadastre_values = []
            for col in ['gnr','bnr','festenr','snr']:
                cadastre_values.append(re.sub(pattern=r'\D',repl='',string=getattr(row,col)))
            #
            GID_values.append(r'/'.join(cadastre_values))
        #
        self.initial_table['GID'] = pd.Series(GID_values,dtype = 'str')
        return self
    #
    def set_tax_year(self) -> Self:
        self.initial_table['Skatteår'] = self.publication_year
        return self
    #
    def set_municip_values(self) -> Self:
        self.initial_table['knr'] = self.knr
        self.initial_table['kommunenavn'] = self.kommunenavn
        return self
    #
    @staticmethod
    def clean_and_convert_float(value: str) -> float|None:
        value = value.replace(',','.')
        value_splits = value.split(".")
        value_parts = [value_part for value_part in value.split(".") if re.search(pattern=r"\D",string=value_part) is None]
        float_value = None
        if len(value_parts) == len(value_splits) and len(value_parts) in [1,2]:
            float_value = float('.'.join(value_parts)) #Gjør nå om til float selv om i utgangspunktet er integer
        # 
        return float_value 
    #
    @staticmethod
    def clean_and_convert_integer(value: str) -> int|None:
        converted_value = None
        #
        if re.search(pattern=r"\D",string=value) is None:
            converted_value = int(value)
        return converted_value  
    #
    @staticmethod
    def clean_and_convert2num(value: str,integer_col: bool = False)  -> int|float|None:
        value = re.sub(pattern=r'\s+',repl='',string=value)
        num_value = None
        # Sjekker om er desimaltall eller integer
        #Sett lik missing hvis ingen meningsfull takstverdi
        #    
        if not integer_col and re.search(r"\d(,|\.)\d",value) is not None:
            num_value = Standard_pdf.clean_and_convert_float(value)
        elif re.search(pattern=r"\d",string = value) is not None:
            num_value = Standard_pdf.clean_and_convert_integer(value)
        #
        return num_value
    #
    @staticmethod
    def conditionally_clean_and_convert2num(value: str|np.int64|int|float,integer_col: bool = False) -> int|float|None:
        cleaned_value = None
        if isinstance(value,int) or isinstance(value,float):
            cleaned_value = value
        elif isinstance(value,np.int64):
            cleaned_value = int(value)     
        elif isinstance(value,str):
            cleaned_value = Standard_pdf.clean_and_convert2num(value,integer_col = integer_col)
        #
        return cleaned_value
    #
    def validate_standard_table(self,standard_table: pd.DataFrame) -> None:
        missing_columns = [col for col in self.relevant_columns if col not in standard_table]
        if len(missing_columns) > 0:
            raise_missing_columns_in_standard_table_exception(missing_columns)
        #
        columns_wrong_datatype = []
        for col in self.relevant_columns:
            if col in self.integer_columns and re.search(pattern= r'^int(\d{2})?$',string= str(standard_table[col].dtype),flags=re.IGNORECASE)  is None:
                columns_wrong_datatype.append(col)
                print(f"Datatypen til kolonnen {col} i standard_table er {str(standard_table[col].dtype)} og skulle vært integer")
            elif col in self.numeric_columns and col not in self.integer_columns and re.search(pattern= r'^float(\d{2})?$',string= str(standard_table[col].dtype),flags=re.IGNORECASE)  is None:
                print(f"Datatypen til kolonnen {col} i standard_table er {str(standard_table[col].dtype)} og skulle vært float ")
                columns_wrong_datatype.append(col)
            elif not col in self.numeric_columns and str(standard_table[col].dtype) != 'object':
                columns_wrong_datatype.append(col)
                print(f"Datatypen til kolonnen {col} i standard_table er {str(standard_table[col].dtype)} og skulle vært string")
            #
        #
        if len(columns_wrong_datatype) > 0:
            raise_wrong_datatypes_standard_table_exception(columns_wrong_datatype)
        #
    #
    def create_standard_table(self) -> Self:
        self.set_header()
        self.set_initial_table()
        self.replace_colnames()
        self.resolve_missing_column_dependencies()
        self.create_missing_columns()
        if not self.cadastre_values_already_set:
            self.set_cadastre_values()
        else:
            self.set_GID_values()
        #
        self.set_tax_year()
        self.set_municip_values()
        self.clean_percent_columns()
        self.digitize_numeric_columns()
        self.clean_jumbled_column_pairs()
        # Forteller nå eksplitt til "casting-metoden" at integerkolonnnene skal være integer
        for col in self.numeric_columns:
            integer_col = (col in self.integer_columns)
            list_values = [Standard_pdf.conditionally_clean_and_convert2num(string_val,integer_col=integer_col) for string_val in self.initial_table[col].to_list()]
            if col in self.integer_columns:                
                self.initial_table[col] = pd.array(list_values, dtype=pd.Int64Dtype())
            else:
                self.initial_table[col] = pd.array(list_values, dtype=pd.Float64Dtype())
            #
        #sta
        standard_table = self.initial_table[self.relevant_columns].copy()
        standard_table.info()
        self.validate_standard_table(standard_table)
        self.standard_table = standard_table        
        return self
    #Okke helt "god praksis" å samtidig manipulere objektverdier og returnere noe annet enn
    def extract_standard_table(self) -> pd.DataFrame:
        return self.standard_table
    #
#

