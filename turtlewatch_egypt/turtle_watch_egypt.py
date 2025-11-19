import os
import re
import shutil
import numpy as np
import pandas as pd
from wildlife_datasets.datasets import utils, WildlifeDataset
from .utils import strip_suffixes

replace_strings = {
    'Ż': 'Z',
    'Ł': 'L',
    'ń': 'n',
    'à': 'a',
    'è': 'e',
    'é': 'e',
    'ě': 'e',
    'ó': 'o',
    'ö': 'o',
    'Ó': 'O',
    '’': '\'',
    '\u200e': '',  # Remove LEFT-TO-RIGHT MARK
    '\u200f': '',  # Remove RIGHT-TO-LEFT MARK
}        

identity_replace = {
    '': np.nan,
    'l': 'unknown',
    'r': 'unknown',
    '42016 ni ei ni': '4-2016 ni ei ni',
    '72016 ni ei baby': '7-2016 ni ei baby',
    '112016 ni baby barn r': '11-2016 ni baby barn',
    '_amal kabir': 'amal kabir',
    'anette': 'annette',
    'asunk1': 'mara',
    'asunk2': 'enea',
    'asunk5': 'fame',
    'asukn8': 'asunk8',
    'barney': 'rileside',
    'carn ei1': 'carn1 ei',
    'dabb1 ei': 'cassiopea',
    'dabb3': 'dabb3 ei',
    'dabb3ei': 'dabb3 ei',
    'dabb6ei': 'momo',
    'dabb6 ei': 'momo',
    'elph2': 'elph2 ei',
    'elph3 ei': 'jamal',
    'egunk3': 'zakaria',
    'egunk8': 'tsunami',
    'egunk10': 'alice',   
    'end june gerryandwout van koot-r 2': 'unknown', 
    'female no id': 'unknown',
    'franncina': 'francina',    
    'gt-0006 karen': 'karen',
    'halhala1': 'halhala 1',
    'her1 ei': 'boma',
    'her2': 'her2 ei',
    'her2ei': 'her2 ei',
    'her3': 'her4 ei',
    'her3ei': 'her4 ei',
    'her3 ei': 'her4 ei',
    'her4': 'her4 ei',
    'her4ei': 'her4 ei',
    'her5ei': 'her5 ei',
    'her6ei': 'her6 ei',
    'her8': 'her8 ei',
    'her hunk1': 'hunk1',
    'her hunk2': 'ansofie',
    'her hunk6': 'hunk6',
    'her hunk10': 'layali',
    'hunk11': 'niko',
    'hunk14': 'chris',
    'hunk15': 'pearl',
    'hunk16': 'norman',
    'hunk17': 'christopher columbus',
    'img': 'unknown',
    'juve maxim': 'maxim',
    'kriya': 'krilya',
    'ladilsav': 'ladislav',
    'maas': 'maas ei 1',
    'maas el1': 'maas ei 1',
    'maasei 1': 'maas ei 1',
    'magherita': 'margherita',
    'no': 'unknown',
    'nosureidher3ei': 'unknown',
    'petra-3': 'petra',
    'perolaf': 'per-olaf',
    'per olaf': 'per-olaf',
    'red sea scuba diving - green turtle': 'unknown',
    'rhan': 'rhian',
    'saskia': 'saskia ch',
    'sata1ei': 'sata1 ei',
    'shab 1': 'shab1 ei',
    'tondoba': 'louisa',
    'tond1ei': 'tond1 ei',
    'tond2ei': 'tond2 ei',
    'toro': 'toro ei',
    'unk1': 'phealyn',
    'unk2': 'sebastian',
    'unk3': 'didier', 
    'unk5': 'petra',
    'unk7': 'jan',
    'unk8': 'sascha',
    'unk10': 'lin',
    'unk14': 'maxim',
    'unk15': 'kariman',
    'unk16': 'andrea',
    'unk19': 'altea',
    'unk20': 'platon',
    'unk21': 'ruud',
    'unk24': 'karim',
    'unk27': 'honey',
    'unk30': 'carmen',
    'unk31': 'lisa',
    'unk34': 'manu',
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

def merge_codes(df, index, individuals):
    xs = []
    for name in df['file']:
        x = code_to_info(name, individuals)[index]
        xs.append(x)
    return list(pd.Series(xs).dropna().unique())

def get_code(xs, name='variables'):
    xs = list(xs)
    for y in ['', 'unknown']:
        if len(xs) > 1 and y in xs:
            xs.remove(y)
    if len(xs) > 1:
        print(f'Multiple {name}: {xs}')                
        return xs[0]
    elif len(xs) == 1:
        return xs[0]
    else:
        return np.nan

def fix_chars(x):
    # TODO: it should check on lower only
    for str1, str2 in replace_strings.items():
        if str1 in x:
            x = x.replace(str1, str2)
    return x

def fix_identity(x, individuals):
    if pd.isnull(x):
        return x
    x = x.lower()
    if x.startswith('no id') or x.startswith('noid'):
        return 'unknown'
    
    ys = [
        x,
        x.split(' ')[0],
        ' '.join(x.split(' ')[:2]),
        ' '.join(x.split(' ')[:-1]),
        x.split('-')[0],
        '-'.join(x.split('-')[:-1]),
        x.split('_')[0],
    ]
    for y in ys:
        if y in individuals:
            return y
        for str1, str2 in identity_replace.items():
            if y == str1 and (str2 == 'unknown' or str2 in individuals):
                return str2
    
    return x

def code_to_info(x, individuals):
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
        leader = x_split[2]
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
        orientation, leader, date, place, hour, author = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        identity, orientation, leader, date, place, hour, author = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan    
    identity = fix_identity(identity, individuals)
    return identity, orientation, leader, date, place, hour, author

def info_to_code(identity, orientation, leader, date, place, hour=None, author=''):
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
    code1 = f'{identity}_{orientation}_{leader}_{year}_{month}_{day}_{place}_{hour1}_{hour2}'.upper()
    code2 = author.title()
    return f'{code1}_{code2}'

def rename_non_ascii(data):
    for idx, df_row in data.iterrows():
        old_file_name = df_row['path_full']
        new_file_name = df_row['path_full']
        new_file_name = fix_chars(new_file_name)
        if old_file_name != new_file_name:
            data.loc[idx, 'path_full'] = new_file_name
            data.loc[idx, 'file'] = os.path.basename(new_file_name)
            data.loc[idx, 'path'] = os.path.dirname(new_file_name)
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
            shutil.copy(old_file_name, new_file_name)

    # Drop possible duplicates because of renaming
    return data.drop_duplicates()

def load_individuals(file_name=None):
    if file_name is None:
        file_name = f'{os.path.dirname(os.path.abspath(__file__))}/individuals.csv'
    individuals = pd.read_csv(file_name)
    individuals = individuals['Common_name'].to_numpy()
    individuals = [fix_chars(x).lower().strip() for x in individuals]
    return [strip_suffixes(x, [' C', ' (DEAD)']) for x in individuals]

class TurtleWatchEgypt_Base(WildlifeDataset):
    individuals = load_individuals()

    def extract_info(self, i):
        return code_to_info(self.df.loc[i, 'path'].split(os.path.sep)[-1], self.individuals)

    def extract_code(self, i):
        df_row = self.df.iloc[i]
        identity = df_row['identity']
        orientation = df_row['label']
        leader = df_row['leader']
        date = df_row['date']
        place = df_row['place'] if not pd.isnull(df_row['place']) else ''
        hour = df_row['hour']
        author = df_row['author'] if not pd.isnull(df_row['author']) else ''
        return info_to_code(identity, orientation, leader, date, place, hour=hour, author=author)


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
        data['identity'] = data['file'].apply(lambda x: code_to_info(x, cls.individuals)[0])
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

        # Preallocate columns to be able to handle strings and nans without warnings
        data['identity'] = pd.Series([np.nan]*len(data), dtype='object')
        data['leader'] = pd.Series([np.nan]*len(data), dtype='object')
        data['place'] = pd.Series([np.nan]*len(data), dtype='object')
        data['author'] = pd.Series([np.nan]*len(data), dtype='object')

        for _, df_encounter in data.groupby('encounter_id'):
            # Extract information from code
            identities = merge_codes(df_encounter, 0, cls.individuals)
            leaders = merge_codes(df_encounter, 2, cls.individuals)
            dates = merge_codes(df_encounter, 3, cls.individuals)
            places = merge_codes(df_encounter, 4, cls.individuals)
            hours = merge_codes(df_encounter, 5, cls.individuals)
            authors = merge_codes(df_encounter, 6, cls.individuals)
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
            data.loc[df_encounter.index, 'leader'] = get_code(leaders, name='leaders')
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

class TurtleWatchEgypt_Master_Heads(TurtleWatchEgypt_Base):
    def _extract(self):
        data = utils.find_images(self.root)
        suffixes = [' B', ' C', ' L', ' R', ' T', ' BR', ' BL', ' RB', ' LB']
        identities = []
        for x in data['file']:
            x = os.path.splitext(x)[0]
            x = x.split('_')[0].split('(')[0].strip()
            while any(x.endswith(ext) for ext in suffixes):
                for ext in suffixes + [x.lower() for x in suffixes]:
                    if x.endswith(ext):
                        x = x.removesuffix(ext).strip()
                        break
            x = fix_chars(x).lower()

            identities.append(x)
        data['identity'] = identities
        data['path'] = data['path'] + os.path.sep + data['file']