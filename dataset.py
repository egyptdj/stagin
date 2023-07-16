import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, maskers, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class DatasetHCPRest(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None, regression=False, num_samples=-1):
        super().__init__()
        self.filename = 'hcp-rest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_dict = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = maskers.NiftiLabelsMasker(image.torch.load_img(self.roi['maps']))
            self.timeseries_dict = {}
            img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'REST')) if f.endswith('nii.gz')]
            img_list.sort()
            for img in tqdm(img_list, ncols=60):
                id = img.split('.')[0]
                timeseries = roi_masker.fit_transform(image.load_img(os.path.join(sourcedir, 'img', 'REST', img)))
                if not len(timeseries) == 1200: continue
                self.timeseries_dict[id] = timeseries
            torch.save(self.timeseries_dict, os.path.join(sourcedir, f'{self.filename}.pth'))

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())

        if 0 < num_samples < len(self.full_subject_list):
            self.full_subject_list = choices(self.full_subject_list, k=num_samples)

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'hcp.csv')).set_index('Subject')

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list
        elif isinstance(k_fold, str):
            self.folds = list(behavioral_df[k_fold].unique())
            self.k_fold = {}
            for fold in self.folds:
                self.k_fold[fold] = [
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]!=fold].index], 
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]==fold].index]
                ]
      
        self.k = None

        self.num_classes = 1 if regression else len(behavioral_df[target_feature].unique())
        self.behavioral_dict = behavioral_df[target_feature].to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]


    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        
        self.k = fold
        if isinstance(fold, int):
            train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        elif isinstance(fold, str):
            train_idx, test_idx = self.k_fold[fold]
            
        if train:
            shuffle(train_idx)
            self.train = True
        else:
            self.train = False
            
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        label = self.behavioral_dict[int(subject)]

        if label=='F':
            label = torch.tensor(0)
        elif label=='M':
            label = torch.tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': label}


class DatasetHCPTask(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, dynamic_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'TASK', task)) if f.endswith('nii.gz')]
                img_list.sort()
                for subject in tqdm(img_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    timeseries = roi_masker.fit_transform(image.load_img(os.path.join(self.sourcedir, 'img', 'TASK', task, subject)))
                    if not len(timeseries)==task_timepoints[task]:
                        print(f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(timeseries)
                    self.label_list.append(task)
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, f'{self.filename}.pth'))

        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        task = self.label_list[self.fold_idx[idx]]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}


class DatasetUKBRest(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='31-0.0', smoothing_fwhm=None, regression=False, num_samples=-1):
        super().__init__()
        self.filename = 'ukb-rest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pkl')):
            with open(os.path.join(sourcedir, f'{self.filename}.pkl'), 'rb') as f:
                self.timeseries_dict = pickle.load(f)
        else:
            self.timeseries_dict = {}
            timeseries_list = [f for f in os.listdir(os.path.join(sourcedir, 'roitimeseries', 'ukb_rest')) if f.endswith('.pth')]
            for timeseries in tqdm(timeseries_list, ncols=60):
                id = timeseries.split('.')[0]
                self.timeseries_dict[id] = torch.load(os.path.join(sourcedir, 'roitimeseries', 'ukb_rest', timeseries))
                if len(timeseries) < 490: continue
            with open(os.path.join(sourcedir, f'{self.filename}.pkl'), 'wb') as f:
                pickle.dump(self.timeseries_dict, f)

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'ukb.csv')).set_index('eid')
        self.behavioral_dict = behavioral_df[target_feature].to_dict()

        for id, timeseries in self.timeseries_dict.items():
            if not len(timeseries) == 490:
                self.full_subject_list.remove(id)
            elif self.behavioral_dict[int(id)] is None:
                self.full_subject_list.remove(id)
            elif np.isnan(self.behavioral_dict[int(id)]):
                self.full_subject_list.remove(id)
         
        if 0 < num_samples < len(self.full_subject_list):
            self.full_subject_list = choices(self.full_subject_list, k=num_samples)

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list
        elif isinstance(k_fold, str):
            self.folds = list(behavioral_df[k_fold].unique())
            self.k_fold = {}
            for fold in self.folds:
                self.k_fold[fold] = [
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]!=fold].index], 
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]==fold].index]
                ]
            
        self.k = None

        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]
        self.num_classes = 1 if regression else len(set(self.full_label_list))


    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        
        self.k = fold
        if isinstance(fold, int):
            train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        elif isinstance(fold, str):
            train_idx, test_idx = self.k_fold[fold]
            
        if train:
            shuffle(train_idx)
            self.train = True
        else:
            self.train = False

        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        label = self.behavioral_dict[int(subject)]

        return {'id': subject, 'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.float32 if self.num_classes==1 else torch.int64)}


class DatasetABIDE(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, dynamic_length=None, target_feature='DX_GROUP', smoothing_fwhm=None):
        super().__init__()

        self.filename = 'abide'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        if os.path.isfile(os.path.join(sourcedir, 'abide', f'{self.filename}.pth')):
            self.timeseries_dict = torch.load(os.path.join(sourcedir, 'abide', f'{self.filename}.pth'))
            behavioral_df = pd.read_csv(os.path.join(sourcedir, 'abide', 'participants.tsv'), delimiter='\t').set_index('subject')
        else:
            abide = datasets.fetch_abide_pcp(os.path.join(sourcedir, 'abide'))
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_dict = {}
            img_list = abide['func_preproc']
            for img in tqdm(img_list, ncols=60):
                id = img.rstrip('_func_preproc.nii.gz')[-5:]
                timeseries = roi_masker.fit_transform(image.load_img(img))
                # if not len(timeseries) == 1200: continue
                self.timeseries_dict[id] = timeseries
            torch.save(self.timeseries_dict, os.path.join(sourcedir, 'abide', f'{self.filename}.pth'))
            behavioral_df = pd.DataFrame(abide['phenotypic']).set_index('subject')
            behavioral_df.to_csv(os.path.join(sourcedir, 'abide', 'participants.tsv'), sep='\t')
            behavioral_df = behavioral_df

        _, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        
        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list
        elif isinstance(k_fold, str):
            self.folds = list(behavioral_df[k_fold].unique())
            self.k_fold = {}
            for fold in self.folds:
                self.k_fold[fold] = [
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]!=fold].index], 
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]==fold].index]
                ]
            
        self.k = None

        self.num_classes = len(behavioral_df[target_feature].unique())
        self.behavioral_dict = behavioral_df[target_feature].to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.full_label_list)

        self.dynamic_length = dynamic_length

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        
        self.k = fold
        if isinstance(fold, int):
            train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        elif isinstance(fold, str):
            train_idx, test_idx = self.k_fold[fold]
            
        if train:
            shuffle(train_idx)
            self.train = True
        else:
            self.train = False
                
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]

        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                assert len(timeseries) >= self.dynamic_length, f'timeseries length {len(timeseries)} is shorter than the dynamic_length {self.dynamic_length}'
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]

        label = self.behavioral_dict[int(subject)]
        label = self.label_encoder.transform([label]).squeeze()

        return {'id': subject, 'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}


class DatasetFMRIPREP(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, task='rest', dynamic_length=None, target_feature='Gender', smoothing_fwhm=None, regression=True, prefix='', num_samples=-1):
        super().__init__()
        assert isinstance(prefix, str)
        self.filename = f'{prefix}-fmriprep-rest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_dict = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_dict = {}
            img_list = [f for f in glob(os.path.join(sourcedir, '**', f'*task-{task}*_space-MNI152NLin2009cAsym_*preproc*.nii.gz'), recursive=True)]
            img_list.sort()
            for img in tqdm(img_list, ncols=60):
                id = img.split('/')[-1].split('_')[0]
                timeseries = roi_masker.fit_transform(image.load_img(img))
                # if not len(timeseries) == 1200: continue
                self.timeseries_dict[id] = timeseries
            torch.save(self.timeseries_dict, os.path.join(sourcedir, f'{self.filename}.pth'))

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())

        if 0 < num_samples < len(self.full_subject_list):
            self.full_subject_list = choices(self.full_subject_list, k=num_samples)

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'participants.tsv'), delimiter='\t').set_index('participant_id')

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list
        elif isinstance(k_fold, str):
            self.folds = list(behavioral_df[k_fold].unique())
            self.k_fold = {}
            for fold in self.folds:
                self.k_fold[fold] = [
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]!=fold].index], 
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold]==fold].index]
                ]
            
        self.k = None

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(behavioral_df[target_feature].unique())
        self.num_classes = 1 if regression else len(behavioral_df[target_feature].unique())
        self.behavioral_dict = behavioral_df[target_feature].to_dict()
        for subject in self.timeseries_dict.keys():
            if not subject in self.behavioral_dict.keys():
                self.full_subject_list.remove(subject)
        self.full_label_list = [self.behavioral_dict[subject] for subject in self.full_subject_list]
        
        self.dynamic_length = dynamic_length
        

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        if isinstance(fold, int):
            train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        elif isinstance(fold, str):
            train_idx, test_idx = self.k_fold[fold]
            
        if train:
            shuffle(train_idx)
            self.train = True
        else:
            self.train = False
         
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        
        if not self.dynamic_length is None:
            if self.train:
                assert len(timeseries) >= self.dynamic_length, f'timeseries length {len(timeseries)} is shorter than the dynamic_length {self.dynamic_length}'        
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]

        label = self.behavioral_dict[subject]
        label = self.label_encoder.transform([label])[0]

        return {'id': subject, 'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}
