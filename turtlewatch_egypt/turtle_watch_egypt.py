import os
import re
import shutil
import numpy as np
import pandas as pd
from wildlife_datasets.datasets import utils, WildlifeDataset

replace_strings = {
    'Ż': 'Z',
    'Ł': 'L',
    'ń': 'n',
    'à': 'a',
    'é': 'e',
    'ě': 'e',
    'ó': 'o',
    'Ó': 'O',
    '’': '\'',
    '\u200e': '',  # Remove LEFT-TO-RIGHT MARK
    '\u200f': '',  # Remove RIGHT-TO-LEFT MARK
}        

identity_replace = {
    '': np.nan,
    'altea injured': 'altea',
    'anette': 'annette',
    'asukn8': 'asunk8',
    'boma r': 'boma',
    'dabb3': 'dabb3 ei',
    'dabb3ei': 'dabb3 ei',
    'dabb6ei': 'dabb6 ei',
    'dabb6 ei': 'momo', # This needs to be after all the fixes
    'elph2': 'elph2 ei',
    'egunk8': 'tsunami',
    'egunk10': 'alice',    
    'female no id': 'unknown',
    'franncina': 'francina',    
    'her2': 'her2 ei',
    'her2ei': 'her2 ei',
    'her3': 'her3 ei',
    'her3ei': 'her3 ei',
    'her3 ei': 'her4 ei', # This needs to be after all the fixes
    'her4': 'her4 ei',
    'her4ei': 'her4 ei',
    'her5ei': 'her5 ei',
    'her5ei l': 'her5 ei',
    'her6ei': 'her6 ei',
    'her8': 'her8 ei',
    'hunk6 c': 'hunk6',
    'hunk6 c2': 'hunk6',
    'hunk6 c3': 'hunk6',
    'hunk6 c4': 'hunk6',
    'hunk6 r': 'hunk6',
    'juve maxim': 'maxim',
    'kariman l': 'kariman',
    'ladilsav': 'ladislav',
    'lanea l': 'lanea',
    'lanea r': 'lanea',
    'lin l': 'lin',
    'lisa r': 'lisa',
    'luna-7m': 'luna',
    'maas': 'maas ei 1',
    'maas el1': 'maas ei 1',
    'maasei 1': 'maas ei 1',
    'magherita': 'margherita',
    'michel l': 'michel',
    'no id': 'unknown',
    'no id-b': 'unknown',
    'no id-e': 'unknown',
    'noid-d': 'unknown',
    'noid': 'unknown',
    'noidmale': 'unknown',
    'noid1': 'unknown',
    'noid2': 'unknown',
    'no id6': 'unknown',
    'noid 6': 'unknown',
    'noid 24': 'unknown',
    'no id24': 'unknown',
    'noid 25': 'unknown',
    'noid36': 'unknown',
    'noid37': 'unknown',
    'noid38': 'unknown',
    'noid39': 'unknown',
    'noid40': 'unknown',
    'noid41': 'unknown',
    'noid42': 'unknown',
    'no idnettondoba': 'unknown',
    'nosureidher3ei': 'unknown',
    'petra-3': 'petra',
    'rhan': 'rhian',
    'ronja r': 'ronja',
    'toro': 'toro ei',
    'unk3 didier': 'didier', 
    'unk7 jan': 'jan',
    'unk11': 'unknown', 
    'unk12': 'unknown', 
    'unk17': 'unknown', 
    'unk20 platon': 'platon',
    'unk25': 'unknown',
    'unk29': 'unknown',
    'unk31': 'unknown', 
    'unk33': 'unknown', 
    'unk36': 'unknown', 
    'unkei': 'unknown',
    'wdla1': 'wdla1 ei',
}

def get_encounter_name(x):
    i = x.lower().rfind('sighting')
    if i == -1:
        return np.nan
    i = x.find(os.path.sep, i)
    if i == -1:
        return x
    else:
        return x[:i]

def merge_codes(df, index):
    xs = []
    for name in df['file']:
        x = code_to_info(name)[index]
        xs.append(x)
    return list(pd.Series(xs).dropna().unique())

def get_code(xs, name='variables'):
    if len(xs) > 1:
        print(f'Multiple {name}: {xs}')                
        return xs[0]
    elif len(xs) == 1:
        return xs[0]
    else:
        return np.nan

def code_to_info(x):
    codes_morning = ['am', 'am1']
    codes_afternoon = ['pm', 'un', 'ng', 'ni']
    codes_nan = ['na', 'gpm']
    codes_time = codes_morning + codes_afternoon + codes_nan

    x = os.path.splitext(x)[0]
    x = x.lower()
    x_split = x.split('_')
    if len(x_split) == 10:
        identity = x_split[0].strip()
        orientation = x_split[1]
        code = x_split[2]
        date = f'{x_split[3]}-{x_split[4]}-{x_split[5]}'
        # Formatting type
        if x_split[6] in codes_time:
            i_place, i_noon, i_hour = 8, 6, 7
        elif x_split[8] in codes_time:
            i_place, i_noon, i_hour = 6, 8, 7
        else:
            i_place, i_noon, i_hour = 6, 7, 8
        place = x_split[i_place]
        if x_split[i_hour] == '':
            hour = np.nan
        else:
            if x_split[i_noon] in codes_morning:
                hour = int(x_split[i_hour].split('.')[0])
            elif x_split[i_noon] in codes_afternoon:
                hour = 12 + np.mod(int(x_split[i_hour]), 12)
            elif x_split[i_noon] in codes_nan:
                hour = np.nan
            else:
                raise ValueError(f'Value {x} is not correct')
        author = x_split[9]
        for x in ['-', '(', '.']:
            author = author.split(x)[0]
        author = re.sub(r'\d+$', '', author).strip()
    elif len(x_split) >= 5:
        identity = x_split[0].strip()
        orientation, code, date, place, hour, author = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        identity, orientation, code, date, place, hour, author = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan    
    for str1, str2 in identity_replace.items():
        if identity == str1:
            identity = str2
    return identity, orientation, code, date, place, hour, author

def info_to_code(identity, orientation, code, date, place, hour=None, author=''):
    year = str(date)[:4]
    month = str(date)[5:7]
    day = str(date)[8:10]
    if pd.isnull(hour):
        hour1 = ''
        hour2 = ''
    else:
        if hour < 12:
            hour1 = 'AM'
            hour2 = int(hour)
        else:
            hour1 = 'PM'
            hour2 = int(hour) - 12
    code1 = f'{identity}_{orientation}_{code}_{year}_{month}_{day}_{place}_{hour1}_{hour2}'.upper()
    code2 = author.title()
    return f'{code1}_{code2}'

def rename_non_ascii(data):
    for idx, df_row in data.iterrows():
        old_file_name = df_row['path_full']
        new_file_name = df_row['path_full']
        # TODO: it should check on lower only
        for str1, str2 in replace_strings.items():
            if str1 in new_file_name:
                new_file_name = new_file_name.replace(str1, str2)
        if old_file_name != new_file_name:
            data.loc[idx, 'path_full'] = new_file_name
            data.loc[idx, 'file'] = os.path.basename(new_file_name)
            data.loc[idx, 'path'] = os.path.dirname(new_file_name)
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
            shutil.copy(old_file_name, new_file_name)

    # Drop possible duplicates because of renaming
    return data.drop_duplicates()


class TurtleWatchEgypt_Base(WildlifeDataset):
    def extract_info(self, i):
        return code_to_info(self.df.loc[i, 'path'].split(os.path.sep)[-1])

    def extract_code(self, i):
        df_row = self.df.iloc[i]
        identity = df_row['identity']
        orientation = df_row['label']
        code = df_row['code']
        date = df_row['date']
        place = df_row['place'] if not pd.isnull(df_row['place']) else ''
        hour = df_row['hour']
        author = df_row['author'] if not pd.isnull(df_row['author']) else ''
        return info_to_code(identity, orientation, code, date, place, hour=hour, author=author)


class TurtleWatchEgypt_AED(TurtleWatchEgypt_Base):
    # TODO: add metadata

    @classmethod
    def _download(cls):
        # TODO: add download
        pass

    @classmethod
    def _extract(cls):
        # TODO: this should be somewhere else probably
        data = utils.find_images('.')

        # Get full file names
        data['path_full'] = data['path'] + os.path.sep + data['file']

        # Rename data with non-ASCII characters
        data = rename_non_ascii(data)

        # Get identity
        data['identity'] = data['file'].apply(lambda x: code_to_info(x)[0])
        data = data[~data['identity'].isnull()]

        # Finalize the dataframe
        data = data.sort_values('file').reset_index(drop=True)
        data = data.drop(['path', 'file'], axis=1)
        data = data.rename({'path_full': 'path'}, axis=1)
        data.to_csv('metadata.csv', index=False)

    def create_catalogue(self):
        df = pd.read_csv(f'{self.root}/metadata.csv')
        df['image_id'] = range(len(df))
        return self.finalize_catalogue(df)
    

class TurtleWatchEgypt_NYED(TurtleWatchEgypt_Base):
    # TODO: add metadata

    @classmethod
    def _download(cls):
        # TODO: add download
        pass

    @classmethod
    def _extract(cls):
        data = utils.find_images('.')

        # Ignoring data starting with '.'    
        idx = data['file'].str.startswith('.')
        data = data[~idx]

        # Get full file names
        data['path_full'] = data['path'] + os.path.sep + data['file']

        # Rename data with non-ASCII characters
        data = rename_non_ascii(data)

        # Ignoring corruping data and adding date
        data['date'] = [utils.get_image_date(x) for x in data['path_full']]
        idx = data['date'] == -1
        if sum(idx) > 0:
            print('Ignoring corrupted files:')
            for i in np.where(idx)[0]:
                print(data['path_full'].iloc[i])
        data = data[~idx]

        # Adding artificial encounters
        data['encounter_name'] = data['path_full'].apply(lambda x: get_encounter_name(x))
        idx = data['encounter_name'].isnull()
        if sum(idx) > 0:
            print('Creating artificial encounters:')
            for i, (folder, df_folder) in enumerate(data[idx].groupby('path')):
                data.loc[df_folder.index, 'encounter_name'] = folder.lower()
                print(folder)

        # Sort data    
        data = data.sort_values('encounter_name')
        data = data.reset_index(drop=True)

        # Get encounter_id
        data['encounter_id'] = (data['encounter_name'] != data['encounter_name'].shift()).cumsum()
        assert data['encounter_id'].nunique() == data['encounter_name'].nunique()

        for _, df_encounter in data.groupby('encounter_id'):
            # Extract information from code
            identities = merge_codes(df_encounter, 0)
            codes = merge_codes(df_encounter, 2)
            dates = merge_codes(df_encounter, 3)
            places = merge_codes(df_encounter, 4)
            hours = merge_codes(df_encounter, 5)
            authors = merge_codes(df_encounter, 6)
            # Add individuals if empty
            if len(identities) == 0:
                for name in df_encounter['path']:
                    for name_split1 in name.split(os.path.sep):
                        name_split2 = name_split1.lower().split(' ')
                        if name_split2[0] == 'sighting' and len(name_split2) >= 3:
                            identities.append(name_split2[2])
                            print(f'Adding identity {name_split2[2]}')
                identities = pd.Series(identities).dropna().unique()
            # Save codes
            data.loc[df_encounter.index, 'identity'] = get_code(identities, name='identities')
            data.loc[df_encounter.index, 'code'] = get_code(codes, name='codes')
            data.loc[df_encounter.index, 'place'] = get_code(places, name='places')
            data.loc[df_encounter.index, 'hour'] = get_code(hours, name='hours')
            data.loc[df_encounter.index, 'author'] = get_code(authors, name='authors')
            # Rewrite the information from EXIF date if available 
            date = get_code(dates, name='dates')
            if not pd.isnull(date):
                data.loc[df_encounter.index, 'date'] = date

        # Finalize the dataframe
        data = data.reset_index(drop=True)
        data = data.drop(['path', 'file', 'encounter_name'], axis=1)
        data = data.rename({'path_full': 'path'}, axis=1)
        data.to_csv('metadata.csv', index=False)

    def create_catalogue(self, load_segmentation=False) -> pd.DataFrame:
        df = pd.read_csv(f'{self.root}/metadata.csv')
        df['image_id'] = range(len(df))
        df.loc[df['identity'].isnull(), 'identity'] = 'unknown'
        if load_segmentation:
            cols = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
            segmentation = pd.read_csv(f'{self.root}/segmentation.csv')
            df = pd.merge(df, segmentation, on='image_id', how='outer')
            df['bbox'] = list(df[cols].to_numpy())
            df = df.drop(cols, axis=1)
            df = df.reset_index(drop=True)            
        df['image_id'] = range(len(df))
        return self.finalize_catalogue(df)