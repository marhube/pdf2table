import pandas as pd
import re
import sys
from typing import Self

class MyClass:
    def __init__(self, value: int) -> None:
        self.value = value

    def increment(self) -> Self:
        self.value += 1
        return self

#
import pandasql as ps  # type: ignore

# Definerer min egen "exception" slik at bare trenger å definere "exception-beskjeden" en gang
class CustomizedException(Exception):
    #Custom exception class
    def __init__(self, message):
        super().__init__(message)

def raise_missing_columns(missing_columns) -> None:
    raise CustomizedException("The folllowing required columns are missing: {missing_columns} ") 
#
def raise_not_a_column(column: str) -> None:
    raise CustomizedException("Column {column} is not a column") 
#
def raise_too_many_iterations(iter:int) -> None:
    raise CustomizedException(f"More than permitted number of iterations.\nEither the size of the table exceeds {iter} or there is a danger of a while-loop being infinite.") 
#   
#
# Memo til slev: "max_rows" er eVeldig sær og kanskje ikke nødvendig sikkerhestsforanstaltning for å unngå evig løkkke
# Memo til selv: Kun av fullstendig esoteriske grunner ("pdf-er med standard format blir kolonnen "Skatt" klint opp i "Fritak" som fører til bestemte "feillesninger")
class GridMiner:
    def __init__(
        self,
        tesseract_page_data: pd.DataFrame,
        skip_lines_top: int = 0,
        skip_lines_bottom: int = 0,
        max_rows: int = 1000,    
        max_columns: int = 1000, 
        hard_coded_regex_filter: list[str] = [] #"Stoppord"
        ) :
          # Initialiserer verdier
        self.tesseract_page_data = tesseract_page_data
        self.validate_dataframe_columns()
        self.skip_lines_top = skip_lines_top
        self.skip_lines_bottom = skip_lines_bottom
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.hard_coded_regex_filter = hard_coded_regex_filter
        #Regner ut "right" og "bottom" og finner radnumre
        self.set_right()
        self.set_bottom()
        self.set_rownum()
    #Har denne funksjonen øverst siden den viser  hvilke kolonner som må være med i "tesseract_page_data"
    def validate_dataframe_columns(self) -> None:
        # Validate that a DataFrame contains all required columns.
        required_columns = ['text','conf','left','width','top','height']
        missing_columns = [col for col in required_columns if col not in self.tesseract_page_data]
        if missing_columns:
            raise_missing_columns(missing_columns)
        #
    #
    def extract_table(self) -> pd.DataFrame:
        print("Er nå inne i extract_table")
        page_table_with_colnum = self.add_colnum().sort_values(["colnum","rownum"])
        values_dict = {}
        columns = [column for column in page_table_with_colnum['column'].unique() if not pd.isna(column)]
        #Initailiser
        for column in columns:
            column_values = self.extract_column_values(page_table_with_colnum,column=column)
            values_dict[column] = column_values
        #
        values_table = pd.DataFrame(values_dict)
        return values_table
    #
    def extract_column_values(self,page_table_with_colnum: pd.DataFrame,column:str|int) -> list[str]:
        filter_col = "column"
        if isinstance(column,int):
            filter_col = "colnum"
        #
        column_parts_query = f"conf > 0 and rownum > 0 and {filter_col} == '{column}'"
        column_parts = page_table_with_colnum.query(column_parts_query).sort_values(['rownum','left'])
        column_values_df = (
            column_parts.groupby(["rownum"])["text"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        #Memo til selv: Imputerer nå bevisst inn "" for missing
        all_rownum_values_df = pd.DataFrame({'rownum' : [rownum for rownum in page_table_with_colnum['rownum'].unique() if rownum > 0]})
        outer_column_values_table = pd.merge(left = all_rownum_values_df,right=column_values_df ,on = "rownum",how = 'left')
        #
        outer_column_values_table['text'] = outer_column_values_table['text'].map(lambda x : "" if pd.isna(x) else x)
        column_values =  outer_column_values_table['text'].to_list()
        return column_values
    #

    def add_colnum(self) -> pd.DataFrame:
        # 
        table_area =  self.extract_table_area().query("conf > 0")
        column_boundaries = self.extract_column_boundaries()
          #Må etterpå Konvertere datatypeene til "Int64" som håndterer missingverdier
        # SQL-like join with a complex condition
        #
        query = """
        SELECT table_area.*,column_boundaries.*
        FROM table_area
        LEFT JOIN column_boundaries 
        ON table_area.left > column_boundaries.left_boundary AND table_area.right < column_boundaries.right_boundary
        """  
        page_data_with_colnum = ps.sqldf(query, locals())
        #"Introduksjon av missing-verdier gjør at int-kolonner i Python blir konvertert til float. Vil heller ha disse som Int64"
        float_columns = [col for col in column_boundaries if str(page_data_with_colnum[col].dtype).startswith('float')]
        #
        for col in float_columns:
            page_data_with_colnum[col] = page_data_with_colnum[col].astype('Int64')
        #
        return page_data_with_colnum
    #
        #   Over til å lage skillelinjer mellom kolonner
    # Memo til selv: Pga en del "stygge" pdf-tabeller med kolonneverdier "klint opp i neste kolonne" 
    #har jeg lagt til funksjonalitet for å "hoppe over vanskelige verdier" ved å spesifisere en liste over regular expressions
    # som "finner" disse vanskelige verdiene.
    # . Denne funksjonaliteten er implementert i "filter_column_boundary_data"
    def extract_column_boundaries(self) -> pd.DataFrame:
        #Memo til selv: Legger til "right" og "bottom"
        table_area =  self.extract_table_area().query("conf > 0")   
        header_area = self.extract_header_area().sort_values("left",ascending=False)
        column_boundary_candidates = self.find_column_boundary_candidates()
         #Initialiserer til tomme lister
        column_names : list[str] = []
        left_values : list[int] = []
        right_values : list[int] = []
        #Fra høyre mot venstre
        previous_left = table_area['right'].max() + 1
        for header_area_row in header_area.itertuples():
            right_values.append(previous_left)
            column_names.append(getattr(header_area_row,"text").strip())
            column_name_left_pos  = getattr(header_area_row,"left")
            next_left_value = max([value for value in column_boundary_candidates if value < column_name_left_pos])
            left_values.append(next_left_value)
            previous_left = next_left_value
        #
        #Må snu listene i riktig rekkefølge
        column_names.reverse()
        left_values.reverse()
        right_values.reverse()
        colnum_values = list(range(len(column_names)))
        column_boundaries = pd.DataFrame(
            {
                'colnum': colnum_values,
                'column':column_names,
                'left_boundary': left_values,
                'right_boundary': right_values
            }            
        )
        #
        return  column_boundaries
    #
    def extract_header_area(self) -> pd.DataFrame:
        table_area = self.extract_table_area()
        header_area = table_area.query("conf > 0 and rownum == 0").sort_values("left")
        return header_area
    #

    def find_column_boundary_candidates(self) -> list[int]:
        column_boundary_candidates = []
        page_data = self.extract_table_area().query("conf > 0")
        unique_left_minus_one_values =  [val -1 for val in page_data['left'].unique()]
        #
        for value in sorted(unique_left_minus_one_values):
            filtered_page_data = self.filter_column_boundary_data(page_data.query(f"conf > 0 and left <= {value} and right >= {value}"))
            if filtered_page_data.shape[0] == 0:
                column_boundary_candidates.append(int(value))
            #
        #
        return column_boundary_candidates
    #
    def filter_column_boundary_data(self,page_data: pd.DataFrame) -> pd.DataFrame:
        filtered_page_data = page_data
        for regex_filter in self.hard_coded_regex_filter:
            boolean_list = [re.search(pattern = regex_filter,string=text_value) is None  for text_value in  page_data['text']]
            filtered_page_data = filtered_page_data.loc[boolean_list,:].copy()
        #
        return filtered_page_data
    #
    #Memo til selv: Bør vurdere om "table_area" bare skal inneholde rader med "conf > 0"
    def extract_table_area(self,reset_row_index: bool = True) -> pd.DataFrame:
        page_data = self.tesseract_page_data.copy()
        rownum_table_min = self.skip_lines_top
        rownum_table_max =  page_data['rownum'].max() - self.skip_lines_bottom
        page_data_table_area =  page_data.query(f"rownum >= {rownum_table_min} and rownum <= {rownum_table_max}")
        if reset_row_index: 
            page_data_table_area = self.reset_rownum(page_data_table_area.copy())
        return page_data_table_area
    #
    @staticmethod
    def reset_rownum(page_data: pd.DataFrame) -> pd.DataFrame:
        first_rownum = page_data['rownum'].min()
        page_data['rownum'] = page_data['rownum'] - first_rownum
        return page_data
    #
    def set_rownum(self) -> Self:
        # Tester
        page_data = self.tesseract_page_data.copy()
        #Meo til selv: row_boundaries blir faktisk både "sett" og brukt av sqlpandas
        row_boundaries = self.extract_row_boundaries() 
        #Må etterpå Konvertere datatypeene til "Int64" som håndterer missingverdier
        # SQL-like join with a complex condition
        query = """
        SELECT page_data.*,row_boundaries.*
        FROM page_data
        LEFT JOIN row_boundaries 
        ON page_data.top > row_boundaries.top_boundary AND page_data.bottom < row_boundaries.bottom_boundary
        """  
        #Setter radnumre
        self.tesseract_page_data = ps.sqldf(query, locals())
        for col in row_boundaries.columns:
            self.tesseract_page_data[col] = self.tesseract_page_data[col].astype('Int64')
        #
        return self
    #
    def extract_row_boundaries(self) -> pd.DataFrame:
        #Mmeo til selv: Legger til "right" og "bottom"
        page_data = self.tesseract_page_data.copy()
        top_values = []
        bottom_values = []
        row_boundary_candidates = self.find_row_boundary_candidates()
        previous_bottom = - 10
        iter = 0 # Sikkerhetsforanstaltning for å unngå evig løkke
        #Memo til selv: Går nå ovenfra og ned
        while iter < self.max_rows:
            remaining_page_data = page_data.query(f"top > {previous_bottom} ")[['top','bottom']]
            if remaining_page_data.shape[0] > 0:    
                top_value = previous_bottom 
                #
                bottom_value =  min([value for value in row_boundary_candidates if value > top_value])
                top_values.append(top_value)
                bottom_values.append(bottom_value)
                previous_bottom = bottom_value
            else:
                break # Har kommet til bunnen av tabellen og er ferdig
            #
            iter = iter + 1
            if iter >= self.max_rows:
                raise_too_many_iterations(iter)
            #
        #
        rownum_values = list(range(len(top_values)))
        row_boundaries = pd.DataFrame({"rownum": rownum_values, "top_boundary": top_values, "bottom_boundary" : bottom_values})
        return row_boundaries
    #
    def find_row_boundary_candidates(self) -> list[int]:
        row_boundary_candidates = []
        #
        page_data = self.tesseract_page_data.query("conf > 0").copy()
        unique_bottom_plus_one_values =  [val +1 for val in page_data['bottom'].unique()]
        for value in sorted(unique_bottom_plus_one_values):
            if page_data.query(f"top <= {value} and bottom >= {value}").shape[0] == 0:
                row_boundary_candidates.append(value)
            #
        #
        return row_boundary_candidates
    #
    def set_bottom(self) -> Self:
        self.tesseract_page_data['bottom'] = self.tesseract_page_data['top'] + self.tesseract_page_data['height']
        return self
    #
    def set_right(self) -> Self:
        self.tesseract_page_data['right'] = self.tesseract_page_data['left'] + self.tesseract_page_data['width']
        return self
    #

   