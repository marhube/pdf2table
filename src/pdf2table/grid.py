
import re
import sys
from math import floor,ceil
#
import pandas as pd
from typing import Self,Dict,Union
from pandas import Index # For å gjøre mypy fornøyd
import pandasql as ps  # type: ignore
#
# Definerer min egen "exception" slik at bare trenger å definere "exception-beskjeden" en gang
class CustomizedException(Exception):
    #Custom exception class
    def __init__(self, message):
        super().__init__(message)

def raise_missing_columns(missing_columns) -> None:
    raise CustomizedException(f"The folllowing required columns are missing: {','.join(missing_columns)} ") 
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
# Krever som "default" at for å tilordnet en "rownum" (radnummer) og et "colnum" (kolnummer) må en strengeverdier må inneholde minst én "alphanumeric"  character og ikke underscore
#Definerer her at "ord" med "tegnhøyde" større enn "max_median_factor * median" og samtidig bredde < "median av bredde" skal regnes som "ekstreme" og ignoreres
# når skillelinjene mellom vertikale linjer skal bestemmes. NB. Dette vil fungere dårlig
#Parameteren "row_grouping_criteria" er tenkt brukt på tilfeller der en rad "egentlig hører sammen med forrige rad". Hvis antall unike kolonner med 
# et gitt radnummer er mindre eller lik "row_grouping_criteria" så blir raden slått sammen med forrige rad
class GridMiner:
    def __init__(
        self,
        tesseract_page_data: pd.DataFrame,
        skip_lines_top: int = 0,
        skip_lines_bottom: int = 0,
        max_rows: int = 1000,    
        max_columns: int = 1000, 
        required_regex: str|list[str] = [r'\w',r'^[^_]+$'],
        max_median_factor: float = 2.5, 
        vertical_span_header: int = 1,
        clean_column_name_regex: str|list[str] = r'\W',
        header_area: pd.DataFrame|None = None,
        merge_multiline_column_names : bool = False,
        row_grouping_criteria : int = 0,
        apply_row_boundary_filter : bool = True

    ) :
        # Initialiserer verdier
        if isinstance(required_regex,str):
            required_regex = [required_regex]
        #
        if isinstance(clean_column_name_regex,str):
            clean_column_name_regex = [clean_column_name_regex]
        #
        self.validate_dataframe_columns(tesseract_page_data)
        self.tesseract_page_data = tesseract_page_data        
        self.skip_lines_top = skip_lines_top
        self.skip_lines_bottom = skip_lines_bottom
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.required_regex_filter : list = required_regex
        self.max_median_factor = max_median_factor
        self.vertical_span_header = vertical_span_header
        self.clean_column_name_regex = clean_column_name_regex
        self.header_area = header_area
        self.merge_multiline_column_names = merge_multiline_column_names
        self.row_grouping_criteria = row_grouping_criteria
        self.apply_row_boundary_filter = apply_row_boundary_filter
        #Regner ut "right" og "bottom" og finner radnumre
        self.set_right()
        self.set_bottom()
        self.set_rownum()
    # Funksjonen "get_required_columns" definerer hvilke kolonner som må være med i tabellen
    @staticmethod
    def get_required_columns() -> list[str]:
        required_columns = ['text','conf','left','width','top','height']
        return required_columns
    #
    @staticmethod
    def validate_dataframe_columns(tesseract_page_data: pd.DataFrame) -> None:
        # Validate that a DataFrame contains all required columns.
        required_columns = GridMiner.get_required_columns()
        ['text','conf','left','width','top','height']
        missing_columns = [col for col in required_columns if col not in tesseract_page_data.columns]
        if missing_columns:
            raise_missing_columns(missing_columns)
        #
    #
    def set_right(self) -> Self:
        self.tesseract_page_data['right'] = self.tesseract_page_data['left'] + self.tesseract_page_data['width']
        return self
    #
    def set_bottom(self) -> Self:
        self.tesseract_page_data['bottom'] = self.tesseract_page_data['top'] + self.tesseract_page_data['height']
        return self
    #Memo til selv: Rotårsaken til at "bounding boxes" (bbox) kan overlappe er at OCR-en en gang i blant kan feile på litt spesielle måter
    #Memo til selv: i "remove_bbox_entries" så må  "bboxes_to_remove" inneholde feltene "top","bottom","left" og "right"
    @staticmethod
    def remove_bbox_entries(page_data: pd.DataFrame,bboxes_to_remove : pd.DataFrame) -> pd.DataFrame:
        query_remove_bboxes = """
            SELECT page_data.*
            FROM  page_data 
            LEFT JOIN bboxes_to_remove  ON
            page_data.top = bboxes_to_remove.top AND page_data.bottom = bboxes_to_remove.bottom AND
            page_data.left = bboxes_to_remove.left AND page_data.right = bboxes_to_remove.right
            WHERE bboxes_to_remove.left IS NULL
            """
        #
        page_data =  ps.sqldf(query_remove_bboxes,locals())
        return page_data
    #
    @staticmethod
    def detect_overlapping_bboxes(left_table: pd.DataFrame,right_table: pd.DataFrame|None = None,include_self: bool = False) -> pd.DataFrame:
        if right_table is None:
            right_table = left_table
        #
        where_not_self = ""
        if not include_self:
            where_not_self = "WHERE left_table.top <> right_table.top OR left_table.bottom <> right_table.bottom OR left_table.left <> right_table.left OR left_table.right <> right_table.right"
        #
        #Lager sql-spørring for å detektere overlappende "bounding boxes" (bbox)
        query_overlapping_boxes = f"""
            SELECT left_table.*
            FROM  left_table 
            INNER JOIN  right_table  ON
            (
            (left_table.right >= right_Table.left and left_table.left <= right_table.left) OR
            (right_table.right >= left_table.left and right_table.left <= left_table.left)
            )  AND
            (
            (left_table.bottom >= right_table.top and left_table.top <= right_table.top) OR
            (right_table.bottom >= left_table.top and right_table.top <= left_table.top)
            )
            {where_not_self}    
            """.strip()  
        #
        overlapping_bboxes = ps.sqldf(query_overlapping_boxes, locals())
        return overlapping_bboxes
    #
    @staticmethod
    def filter_overlapping_bboxes(page_data: pd.DataFrame) -> pd.DataFrame:
        overlapping_boxes =  GridMiner.detect_overlapping_bboxes(page_data,include_self = False)
        #Spørring som for hver "bbox" teller for mange andre "bboxer" den overlapper
        query_overlapping_boxes_group_by = """
            SELECT overlap.top,overlap.bottom,overlap.left,overlap.right,COUNT(*) AS n_overlapping_boxes
            FROM  overlapping_boxes overlap
            GROUP BY overlap.top,overlap.bottom,overlap.left,overlap.right
            ORDER BY n_overlapping_boxes DESC,(overlap.bottom-overlap.top) DESC
            """
        #
        while overlapping_boxes.shape[0] > 0:
            #Plukker ut den boksen som overlapper med flest andre bokser og fjerner denne verdien
            overlapping_boxes_top_row =  ps.sqldf(query_overlapping_boxes_group_by).head(1)
            page_data = GridMiner.remove_bbox_entries(page_data,bboxes_to_remove = overlapping_boxes_top_row)
            overlapping_boxes =  GridMiner.detect_overlapping_bboxes(page_data,include_self = False)
        #
        return page_data
    #Mmeo til selv: I "detect_horizontally_overlapping_bboxes_on_same_row" så  kan man med parameteren "include_self" velge
    # om "overlapping" skal inkludere bboxen selv eller ikke. Default er ikke å inkludere
    @staticmethod 
    def detect_horizontally_overlapping_bboxes_on_same_row(left_table: pd.DataFrame,right_table: pd.DataFrame|None = None,include_self: bool = False) -> pd.DataFrame:
        if right_table is None:
            right_table = left_table
        #
        where_not_self = ""
        if not include_self:
            where_not_self = "WHERE left_table.left <> right_table.left OR left_table.right <> right_table.right"
        #
        query_horizontally_overlapping_same_row = f"""
            SELECT left_table.*
            FROM  left_table 
            INNER JOIN  right_table  ON
            left_table.rownum = right_table.rownum  AND 
            (
            (left_table.right >= right_table.left and left_table.left <= right_table.left) OR
            (right_table.right >= left_table.left and right_table.left <= left_table.left) 
            )
            {where_not_self}           
            """.strip()  
        #
        horizontally_overlapping =  ps.sqldf(query_horizontally_overlapping_same_row, locals())
        return horizontally_overlapping 
    #Memo til selv: I filter_horizontally_overlapping_on_same_row så kan man med "remove_all_overlapping_bboxes"  
    # velge om man vil fjerne  all bboxer som overlapper mhp "left" and "right" og radnummer med minst en annen, eller om man
    #vil beholde de som er igjen når man har fjernet "bboxen som overlapper med flest andre først"  inntil det ikke er flere igjen som overlapper
    #med hverandre (slik som i funksjonen "filter_overlapping_bboxe"). For å finne "column_boundaries" har jeg funnet ut at det er best
    # å ta bort alle som i utgangspunktet overlapper
    @staticmethod
    def filter_horizontally_overlapping_on_same_row(page_data: pd.DataFrame,remove_all_overlapping_bboxes : bool = True) -> pd.DataFrame:
        horizontally_overlapping =  GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(page_data,include_self = False)
        #
        query_horizontally_overlapping_same_row_group_by = """
            SELECT overlap.top,overlap.bottom,overlap.left,overlap.right,overlap.text,COUNT(*) AS n_overlapping
            FROM  horizontally_overlapping overlap
            GROUP BY overlap.top,overlap.bottom,overlap.left,overlap.right,overlap.text
            ORDER BY n_overlapping DESC,(overlap.right-overlap.left) DESC
            """
        #
        while horizontally_overlapping.shape[0] > 0:
            #Plukker ut den boksen som overlapper med flest andre bokser og fjerner denne verdien
            horizontally_overlapping_top_row =  ps.sqldf(query_horizontally_overlapping_same_row_group_by).head(1)
            if remove_all_overlapping_bboxes:
                page_data = GridMiner.remove_bbox_entries(page_data,bboxes_to_remove = horizontally_overlapping)   
            else:
                page_data = GridMiner.remove_bbox_entries(page_data,bboxes_to_remove = horizontally_overlapping_top_row)
            #
            horizontally_overlapping = GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(page_data,include_self = False) 
         #
        return page_data            
    #Memo til selv: Bør vurdere om det fortsatt er nødvendig å nyttig å filtrere ut i fra "mediankriteriet" selv etter å ha fjernet overlappende "bounding boxes"
    def apply_median_filter(self,page_data: pd.DataFrame) -> pd.DataFrame:
        #Fjerner først eventuelle missingverdier i text
        #Bruker nå et "numerisk basert filter" basert
        median_height = self.tesseract_page_data.query("conf > 0")['height'].median()
        median_width = self.tesseract_page_data.query("conf > 0")['width'].median()
        filtered_page_data = page_data.query(f"height < {self.max_median_factor * median_height} or width > {median_width}").copy() 
        return filtered_page_data
    #   
    def apply_regex_filter(self,page_data: pd.DataFrame) -> pd.DataFrame:
        filtered_page_data = page_data.query("conf > 0").copy()
        for regex in self.required_regex_filter:
            boolean_list = [isinstance(text_value,str) and re.search(pattern = regex,string=text_value) is not None  for text_value in  filtered_page_data['text']]
            filtered_page_data = filtered_page_data.loc[boolean_list,:].copy()
        #
        return filtered_page_data
    #
    def filter_boundary_data(self,page_data: pd.DataFrame,row_boundaries : bool = True) -> pd.DataFrame:  
        if row_boundaries:
            page_data = GridMiner.filter_overlapping_bboxes(page_data)
        else:
            page_data = GridMiner.filter_horizontally_overlapping_on_same_row(page_data)
        #
        filtered_page_data = self.apply_median_filter(page_data)        
        filtered_page_data = self.apply_regex_filter(filtered_page_data)
        #
        return filtered_page_data
    #
    def find_row_boundary_candidates(self) -> list[int]:
        row_boundary_candidates = []
        #Memo til selv: Filtrerer nå bevistt ikke kandidater
        page_data = self.tesseract_page_data.query("conf > 0").copy()
        #
        filtered_page_data  = page_data.copy()
        if self.apply_row_boundary_filter:
            filtered_page_data = self.filter_boundary_data(filtered_page_data,row_boundaries = True) 
        #OB
        unique_bottom_plus_one_values =  [val +1 for val in page_data['bottom'].unique()]
        for value in sorted(unique_bottom_plus_one_values):
            if  filtered_page_data.query(f"top <= {value} and bottom >= {value}").shape[0] == 0:
                row_boundary_candidates.append(value)
            #
        #
        return row_boundary_candidates
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
    def set_rownum(self) -> Self:
        # Tester
        page_data = self.tesseract_page_data.copy()
        #Memo til selv: page_data og row_boundaries blir faktisk både "sett" og brukt av sqlpandas
        # Det er bevisst at det er "page_data" som er "left table"
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
    @staticmethod
    def reset_rownum(page_data: pd.DataFrame) -> pd.DataFrame:
        page_data = page_data.copy()
        first_rownum = page_data['rownum'].min()
        page_data['rownum'] = page_data['rownum'] - first_rownum        
        return page_data
    #
    def put_header_on_same_row(self,page_data: pd.DataFrame) -> pd.DataFrame:
        if self.vertical_span_header > 1:
            page_data['rownum'] = page_data['rownum'].map(lambda x: 0 if x>= 0 and  x < self.vertical_span_header else x)
        #
        return page_data
    #Memo til selv: Bør vurdere om "table_area" bare skal inneholde rader med "conf > 0"
    def extract_non_void_lines(self,reset_rownum: bool = False) -> pd.DataFrame:
        page_data = self.tesseract_page_data.copy()
        filtered_page_data = page_data.copy()
        filtered_page_data = self.apply_regex_filter(filtered_page_data)
        filtered_page_data = self.apply_median_filter(filtered_page_data)
        unique_rownum_filtered_page_data =  [int(rownum_value) for rownum_value in  filtered_page_data['rownum'].unique() if not pd.isna(rownum_value)]
        non_void_lines = filtered_page_data.query(f"rownum in {unique_rownum_filtered_page_data}")
        if reset_rownum:
            non_void_lines = self.reset_rownum(non_void_lines)
        #
        return non_void_lines
    #
    #Memo til selv: "create_compunded_bbox" er en kjempeesoterisk hjelpefunksjon til "create_header_area_with_multiline_column_names" som tar en liste
    # av "bboxes" og slår dem sammen til en ny stor "bbox" som inneholder alle. Tenkt bruk er i forbindelse med å slå sammen overlappende "bboxes"
    @staticmethod 
    def create_aggregated_bbox(bbox_data: pd.DataFrame) ->  Dict[str, Union[int, str]]:
        # Compute the aggregated coordinates as ints first.
        left_val = int(bbox_data['left'].min())
        top_val = int(bbox_data['top'].min())
        right_val = int(bbox_data['right'].max())
        bottom_val = int(bbox_data['bottom'].max())
        #
        # Build the dictionary with the known types.
        aggregated_bbox_dict: Dict[str, Union[int, str]] = {
            "left": left_val,
            "top": top_val,
            "right": right_val,
            "bottom": bottom_val,
            "width" : right_val - left_val,
            "height": bottom_val - top_val,
            "text": "\n".join(bbox_data['text'].to_list()),
        }
        #
        return aggregated_bbox_dict
    #  Memo til selv: I funksjonen "create_header_area_with_multiline_column_names" så betyr "recursive" at søkingen etter "horisontalt overlappende bboxes" er "transitiv", dvs.
    #    at hvis "bbox A" overlapper med "bbox B" og "bbox B" overlapper med "bbox C" så lages det en ny bbox som inkluderer både A, B og C.
    @staticmethod
    def create_header_area_with_multiline_column_names(header_area: pd.DataFrame,recursive : bool = True) -> pd.DataFrame:
        header_area_copy = header_area.copy()
        #Initierer "new_header_area" som en tom dataramme med de samme kolonnene som "header_area"
        new_header_area =  header_area_copy.iloc[0:0,:].copy()
        #
        n_bboxes_detected = 0
        while header_area_copy.shape[0] > 0:
            next_top_row = header_area_copy.iloc[0:1,:].copy() #Trekker ut øverste rad
            continue_search = True
            n_bboxes_detected = 0
            while continue_search:
                next_overlapping_values = GridMiner.detect_horizontally_overlapping_bboxes_on_same_row(left_table = header_area_copy,right_table= next_top_row,include_self = True).sort_values(by=["top"])
                aggegated_bbox_dict = GridMiner.create_aggregated_bbox(next_overlapping_values)
                for key in aggegated_bbox_dict.keys():
                    next_top_row[key] = aggegated_bbox_dict[key]
                #                                
                #Fjerner rader som vi er ferdige med
                if not recursive or next_overlapping_values.shape[0] <= n_bboxes_detected:
                    #Når jeg kommer inn i denne if-en så er vi ferdige med å samle opp alle overlappende bboxer og skal gå gå videre til neste "top_row"
                    # Før det så skal de overlappende bboxene som er funnet fjernes fr
                    continue_search = False                    
                    header_area_copy = GridMiner.remove_bbox_entries(header_area_copy,bboxes_to_remove = next_overlapping_values)                     
                #Må oppdaterer "n_bboxes_dete"  
                n_bboxes_detected = next_overlapping_values.shape[0]               
            #
            new_header_area = pd.concat([new_header_area,next_top_row]) #Memo til selv: OBS: Gjøres først etter at at ute av den innerste while-løkken
        #
        return new_header_area
    #
    def gather_multiline_column_names(self,page_data_table_area: pd.DataFrame) -> pd.DataFrame:
        header_area = page_data_table_area.query("conf > 0 and rownum == 0")
        table_values = page_data_table_area.query("conf <= 0  or  rownum != 0")
        new_header_area = self.create_header_area_with_multiline_column_names(header_area)
        new_table_area = pd.concat([new_header_area,table_values])
        return new_table_area
    #
    # Memo til selv: Bør vurdere om skal skille ut den første filtreringen i "extract_table_area" i en egen funksjon 
    def extract_table_area(self) -> pd.DataFrame:
        non_void_lines = self.extract_non_void_lines(reset_rownum=True)
        rownum_table_max =   non_void_lines['rownum'].max() - self.skip_lines_bottom
        page_data_table_area =  non_void_lines.query(f"rownum >= {self.skip_lines_top} and rownum <= {rownum_table_max}").copy()
        page_data_table_area = self.reset_rownum(page_data_table_area)
        page_data_table_area = self.put_header_on_same_row(page_data_table_area)
        #Må håndtere tilfellet der det ikke er noen kolonneoverskrift. I så fall så er "rownum 0", ikke en overskrift, men en del av selve tabellen.
        if self.vertical_span_header == 0:
            page_data_table_area['rownum'] = page_data_table_area['rownum'].map(lambda x: x+1)
        #Legger til "header_area" hvis denne er "preutfylt"
        if isinstance(self.header_area,pd.DataFrame):
            page_data_table_area = pd.concat([self.header_area,page_data_table_area])
        #
        if self.merge_multiline_column_names:
            page_data_table_area = self.gather_multiline_column_names(page_data_table_area)
        #            
        return page_data_table_area
    #
    #   Over til å lage skillelinjer mellom kolonner
    # Memo til selv: Pga en del "stygge" pdf-tabeller med kolonneverdier "klint opp i neste kolonne" 
    #har jeg lagt til funksjonalitet for å "hoppe over vanskelige verdier" ved å spesifisere en liste over regular expressions
    # som "finner" disse vanskelige verdiene.
    # . Denne funksjonaliteten er implementert i "filter_column_boundary_data"
   # Memo til selv: Pga mulig "kræsj"/overlapping i horizontal posisjon mellom kolonneoverskriftene så filtrerer jeg også nå på
   # at det bare er verdiene i selve tabellen som ikke skal overskride "kolonnegrensen", mens det ikke gjør noe hvis selve kolonneoverskriftene gjør det
   #Et god eksemepel på en slik "kræsj" mellom kolonneoverskriftene er pdf-en fra Øystre Slidre (som generelt er veldig styggs)
   # I de tilfellene der kolonneoverskriftene (kolonnenavenene) strekker seg over flere linjer så ignores disse kolonnenavnene når det skal defineres
   # hvor skillelinjene mellom de ulike kolonnene skal gå, men ellers ikke
    #filter_overlapping_headers er en ekstremt sær/esoterisk funksjon!!!
    #
    def find_column_boundary_candidates(self) -> list[int]:
        column_boundary_candidates = []
        table_area = self.extract_table_area().query("conf > 0")
        filtered_page_data = self.filter_boundary_data(table_area,row_boundaries = False) 
        #Memo til selv: OBSSSSSSS Må passe på å velge kandiater fra den ufiltrerte "table_area" og ikke fra "filtered_page_data"
        # Dette fordi koloneneoverskriftene (f.eks for Vefsn) i noen tilfeller er spredt over to linjer med overlappende "left" og "right"
        unique_left_minus_one_values =  [int(val) -1 for val in table_area['left'].unique()]
        for value in sorted(unique_left_minus_one_values):
            #Memo til selv: Tar nå kun bort akkurat de kolonnenavnene jeg må ta bort
            if  filtered_page_data.query(f"rownum >= 0 and left <= {value} and right >= {value}").shape[0] == 0:
                column_boundary_candidates.append(value)
            #
        #
        return column_boundary_candidates
    #
    def extract_header_area(self) -> pd.DataFrame:
        table_area = self.extract_table_area()
        header_area = table_area.query(f"conf > 0 and rownum == 0").sort_values("left")
        return header_area
    #
    def clean_column_name(self,column_name: str) -> str:
        for regex in self.clean_column_name_regex:
            column_name = re.sub(pattern=regex,repl='',string=column_name)
        #
        return column_name
    #
    def extract_column_boundaries(self) -> pd.DataFrame:
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
            column_name = self.clean_column_name(getattr(header_area_row,"text").strip())
            column_names.append(column_name)
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
    def add_colnum(self) -> pd.DataFrame:
        # 
        table_area =  self.extract_table_area().query("conf > 0")
        column_boundaries = self.extract_column_boundaries()
          #Må etterpå Konvertere datatypeene til "Int64" som håndterer missingverdier
        # SQL-like join with a complex condition
        #Memo til selv: OBBS Har nå "column boundaries som "left-table". På den måten sikres at alle kolonnene
        #"kommer med"
        query = """
        SELECT table_area.*,column_boundaries.*
        FROM  column_boundaries  
        LEFT JOIN  table_area
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
    # Selv om dataene i samme "celle" har samme radnummer så er det i noen tilfller en eller flere av delene i cellen som ligger
    #over (eller under) andre deler slik at man kan trekke en skillelinje mellom dem.
    #Memo til selv: OBSS Det er mulig at "add_intra_cell_rownum" feiler hvis "column_data" er en tom tabell uten rader.
    #Memo til selv: Denne funksjoen feiler av en eller annen grunn which "column_data" har 0 rader
    def add_intra_cell_rownum(self,column_data: pd.DataFrame) -> pd.DataFrame:
        #Memo til selv: Må håndtere tilfeller med 0 rader
        #Initierer som en dataramme med 0 rader og de samme kolonnene som "column_data" 0 "intra_cell_rownum"
        list_cell_data_with_intra_cell_rownum = []
        cell_data_with_intra_cell_rownum = column_data.iloc[0:0,:].copy()
        cell_data_with_intra_cell_rownum['intra_cell_rownum'] = 0 
        #
        for rownum in sorted(set(column_data['rownum'])):
            #Plukker ut "celledata" (data i samme kolonne med samme radnummer). 
            cell_data = column_data.query(f"conf > 0 and rownum == {rownum}").copy()
            cell_data.columns = Index(['outer_rownum' if col == 'rownum' else col for col in cell_data.columns])
            #Lager "interne radnumre" som deler opp innholdet i gjeldende cell inn i flere linjer hvis noen verdiene i cellen  er vertikalt adskilt fra de andre
            cell_grid_obj = GridMiner(tesseract_page_data = cell_data[['text','conf','left','width','top','height','outer_rownum']].copy(),apply_row_boundary_filter = False)
            cell_tesseract = cell_grid_obj.tesseract_page_data 
            #Må bytte tilbake til ønskede radnavn
            replacement_dict = {'rownum' : 'intra_cell_rownum', 'outer_rownum' : 'rownum' }
            cell_tesseract.columns = Index([replacement_dict[col] if col in replacement_dict.keys() else col for col in cell_tesseract.columns])
            list_cell_data_with_intra_cell_rownum.append(cell_tesseract)
            #
        #
        if len(list_cell_data_with_intra_cell_rownum) > 0:
            cell_data_with_intra_cell_rownum = pd.concat(list_cell_data_with_intra_cell_rownum)        
        #  
        return cell_data_with_intra_cell_rownum 
    #
    def extract_column_values(self,page_table_with_colnum: pd.DataFrame,column:str|int) -> list[str]:
        filter_col = "column"
        if isinstance(column,int):
            filter_col = "colnum"
        #
        column_parts_query = f"conf > 0 and rownum > 0 and {filter_col} == '{column}'"
        column_parts = page_table_with_colnum.query(column_parts_query).copy()
        #Legger til "intracellerad" for å takle de tilfeller der verdier i samme celle strekker seg over mer enn én linje
        #Kan bare sortere
        #
        column_parts_with_intra_cell_rownum = self.add_intra_cell_rownum(column_parts).sort_values(['rownum','intra_cell_rownum','left'])
        #
        column_values_df = (
            column_parts_with_intra_cell_rownum .groupby(["rownum"])["text"]
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
    #Slår sammen rader som tilhører samme "celle" i de tilfellen der antall unike kolonner med en verdi på denne raden er midre eller lik
    # self.row_grouping_criteria  Gjør dette "rekursivt", fortsetter til finner første  slik rad, endrer radnummer på denne og søker så på nytt
    # til ikke finner flere slike tilfeller
    #
    def gather_multiline_cells(self,page_data : pd.DataFrame) -> pd.DataFrame:
        if self.row_grouping_criteria >= 1:
            done_grouping = False
            while not done_grouping:
                unique_rownums_larger_than_one = [rownum for rownum in page_data['rownum'].unique() if rownum >= 2]
                done_grouping = True
                for rownum in  unique_rownums_larger_than_one:
                    count_unique_colnums = len(page_data.query(f"rownum == {rownum}")['colnum'].unique())
                    if count_unique_colnums <= self.row_grouping_criteria:            
                        page_data['rownum'] = page_data['rownum'].map(lambda x: x-1  if x == rownum  else  x)
                        done_grouping = False          
                        break
                    #
                #
            #                
        #      
        return page_data
    #
    def extract_table(self) -> pd.DataFrame:
        #page_data = self.tesseract_page_data.copy()
        page_table_with_colnum = self.add_colnum().sort_values(["colnum","rownum"])
        page_table_with_colnum = self.gather_multiline_cells(page_table_with_colnum)
        #
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
