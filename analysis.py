import os
import random
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from util import bold, option
from dataset import DatasetHCPRest, DatasetUKBRest
from torch import save, load
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve
from nilearn.image import load_img, new_img_like
from nilearn.datasets import fetch_atlas_schaefer_2018
from nipy.modalities.fmri.glm import GeneralLinearModel
from pingouin import chi2_independence


def node_attention_glm(node_attention, design_matrix, contrast):
    # [time, node] / [time, task] / [task]
    assert len(node_attention) == len(design_matrix)
    assert len(design_matrix.T) == len(contrast)
    model = GeneralLinearModel(design_matrix)
    model.fit(node_attention)
    return model.contrast(contrast)


def time_attention_threshold(time_attention, std=1.0, consecutive_timepoint=3):
    # [source, target]
    target_attention = time_attention.mean(axis=0)
    threshold = target_attention.mean() + std * target_attention.std()
    significant_timepoints = (target_attention>threshold).nonzero()[0]
    if not len(significant_timepoints) == 0:
        significant_timepoints = [cluster[len(cluster)//2] for cluster in np.split(significant_timepoints, np.where(np.diff(significant_timepoints) > 1+consecutive_timepoint)[0]+1)]
    return significant_timepoints


def plot_nifti(node_attention, roi, analysisdir, topk=0, prefix='', merge=False):
    roi_array = roi.get_fdata()
    assert len(node_attention)==roi_array.max()

    order = np.argsort(node_attention)[::-1]
    node_attention = node_attention[order]
    if topk:
        node_attention = node_attention[:topk]
        order = order[:topk]

    if merge:
        lh_roi_attention_array = np.zeros_like(roi_array)
        rh_roi_attention_array = np.zeros_like(roi_array)
        for k, (i, attention) in enumerate(zip(order, node_attention)):
            if i<200:
                lh_roi_attention_array[roi_array==(i+1)] = attention
            else:
                rh_roi_attention_array[roi_array==(i+1)] = attention
        new_img_like(roi, lh_roi_attention_array, roi.affine).to_filename(os.path.join(analysisdir, prefix, f'top{topk}_merge_lh.nii.gz'))
        new_img_like(roi, rh_roi_attention_array, roi.affine).to_filename(os.path.join(analysisdir, prefix, f'top{topk}_merge_rh.nii.gz'))

    else:
        for k, (i, attention) in enumerate(zip(order, node_attention)):
            roi_attention_array = np.zeros_like(roi_array)
            roi_attention_array[roi_array==(i+1)] = attention
            new_img_like(roi, roi_attention_array, roi.affine).to_filename(os.path.join(analysisdir, prefix, f'top{k+1}_roi{i+1}.nii.gz'))


def analyze_rest(argv):
    assert argv.roi=='schaefer', 'Schafer400 atlas is currently supported for analysis'
    analysisdir = os.path.join(argv.targetdir, 'analysis')
    roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(argv.sourcedir, 'roi'))
    roidir = roi['maps']
    roimetadir = os.path.join(argv.sourcedir, 'roi', '7_400_coord.csv')

    # summarize roi
    roimeta = pd.read_csv(roimetadir, index_col=0, delimiter=',')
    roimeta = roimeta.loc[1:]
    roidict = {}
    for i, (index, meta) in enumerate(roimeta.iterrows()):
        labels = meta['Label Name'].split('_')[1:-1]
        if len(labels)==2:
            labels.append(labels[-1])
        roidict[i+1] = {'index': index, 'hemisphere': labels[0], 'network': labels[1], 'region': labels[2], 'R': meta['R'], 'A': meta['A'], 'S': meta['S']}

    os.makedirs(analysisdir, exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'node_attention'), exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'time_attention'), exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'figures'), exist_ok=True)

    samples = load(os.path.join(argv.targetdir, 'samples.pkl'))
    node_attention = np.concatenate([np.load(os.path.join(argv.targetdir, 'attention', str(k), 'node_attention.npy')) for k in range(argv.k_fold)])
    time_attention = np.concatenate([np.load(os.path.join(argv.targetdir, 'attention', str(k), 'time_attention.npy')) for k in range(argv.k_fold)])
    label = np.concatenate([samples['true'][k] for k in range(argv.k_fold)])

    roi_index = {'L.VN':0, 'L.SMN':31, 'L.DAN':68, 'L.SVN': 91, 'L.LN': 113, 'L.CCN': 126, 'L.DMN': 148,
                    'R.VN':200, 'R.SMN':230, 'R.DAN':270, 'R.SVN': 293, 'R.LN': 318, 'R.CCN': 331, 'R.DMN': 361}
    roi_range = {'L.VN':[0,31], 'L.SMN':[31,68], 'L.DAN':[68,91], 'L.SVN': [91,113], 'L.LN': [113,126], 'L.CCN': [126,148], 'L.DMN': [148,200],
                    'R.VN':[200,230], 'R.SMN':[230,270], 'R.DAN':[270,293], 'R.SVN': [293,318], 'R.LN': [318,331], 'R.CCN': [331,361], 'R.DMN': [361,400]}
    roi_tick_index = {'L.VN':15, 'L.SMN':49, 'L.DAN': 79, 'L.SVN': 102, 'L.LN': 119, 'L.CCN': 137, 'L.DMN': 174,
                        'R.VN':215, 'R.SMN':250, 'R.DAN': 281, 'R.SVN': 305, 'R.LN': 324, 'R.CCN': 346, 'R.DMN': 380}
    dmn_index = {'L.Temp':0, 'L.PFC':17, 'L.PCC': 17+24, 'R.Par': 17+24+11, 'R.Temp':17+24+11+5, 'R.PFCv':17+24+11+5+8, 'R.PFCm': 17+24+11+5+8+4, 'R.PCC': 17+24+11+5+8+4+13}
    dmn_tick_index = {'L.Temp':(0+17)//2, 'L.PFC':(17+17+24)//2, 'L.PCC': (17+24+17+24+11)//2, 'R.Par': (17+24+11+17+24+11+5)//2, 'R.Temp':(17+24+11+5+17+24+11+5+8)//2, 'R.PFCv':(17+24+11+5+8+17+24+11+5+8+4)//2, 'R.PFCm': (17+24+11+5+8+4+17+24+11+5+8+4+13)//2, 'R.PCC': (17+24+11+5+8+4+13+17+24+11+5+8+4+13+9)//2}

    sns.set_theme(context='paper', style="white", font='Freesans', font_scale=2.0, palette='muted')

    print('plot exemplar time attention')
    os.makedirs(os.path.join(analysisdir, 'time_attention', 'examples'), exist_ok=True)
    for i, attention in enumerate(time_attention[::100]):
        fig, ax = plt.subplots(4,2, figsize=(15,16), gridspec_kw={'height_ratios':[1,10,1,10]}, sharex='col')
        fig.tight_layout()
        for layer, layer_attention in enumerate(attention):
            threshold = time_attention_threshold(layer_attention)
            sns.lineplot(x=np.arange(len(layer_attention)), y=layer_attention.mean(0), linewidth=2.0, ax=ax[2*(layer//2)][layer%2])
            sns.lineplot(x=np.arange(len(layer_attention)), y=layer_attention.mean()+1*layer_attention.mean(0).std(), linewidth=1.0, linestyle='--', ax=ax[2*(layer//2)][layer%2])
            ax[2*(layer//2)][layer%2].plot(threshold, 1.2*layer_attention.mean(0).max()*np.ones_like(threshold), linestyle='None', marker='o', color='red', clip_on=False)
            ax[2*(layer//2)][layer%2].set_title(f'Layer {layer+1}')
            sns.heatmap(data=layer_attention, xticklabels=60, yticklabels=60, cbar=False, ax=ax[2*(layer//2)+1][layer%2])
        plt.savefig(os.path.join(analysisdir, 'time_attention', 'examples', f'rest_time_attention{i}.png'))
        plt.close()
    
    print('plot exemplar node attention')
    os.makedirs(os.path.join(analysisdir, 'node_attention', 'examples'), exist_ok=True)
    for i, layer_attention in enumerate(node_attention[::100]):
        fig, ax = plt.subplots(2, 3, figsize=(18,15), sharey='row', gridspec_kw={'width_ratios':[25,25,2]})
        gs = ax[0][2].get_gridspec()
        for _ax in ax[:,2]:
            _ax.remove()
        cbar_ax = fig.add_subplot(gs[:,2])
        for layer, attention in enumerate(layer_attention):
            sns.heatmap(ax=ax[layer//2][layer%2], vmin=0.0, vmax=1.0, data=attention, xticklabels=50, yticklabels=60, cbar=True if layer==0 else False, cbar_ax=cbar_ax)
            ax[layer//2][layer%2].set_title(f'Layer {layer+1}')
            ax[layer//2][layer%2].tick_params(direction='out', length=5, width=2)
            ax[layer//2][layer%2].set_xticks(list(roi_tick_index.values()))
            ax[layer//2][layer%2].set_xticklabels(list(roi_tick_index.keys()), rotation=45, fontsize=10)
            ax[layer//2][layer%2].vlines(list(roi_index.values()), *ax[layer//2][layer%2].get_ylim(), colors='black', linestyle='--', linewidth=1, alpha=1.0)
        plt.savefig(os.path.join(analysisdir, 'node_attention', 'examples', f'rest_node_attention{i}.png'))
        plt.close()

    try:
        model = joblib.load(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'model', f'model_subsample{argv.subsample}.joblib'))
    except:
        print('prepare subsampled adjacency and train kmeans model')
        os.makedirs(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering'), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'model'), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure'), exist_ok=True)
        if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold)
        elif argv.dataset=='ukb-rest': dataset = DatasetUKBRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature)
        else: raise
        subsample_a_list = []
        for k in range(argv.k_fold):
            dataset.set_fold(k, train=False)
            for data in tqdm(dataset, ncols=60):
                dyn_a, _ = bold.process_dynamic_fc(data['timeseries'].unsqueeze(0), argv.window_size, argv.window_stride)
                subsample_a_list.append(np.stack([(a > np.percentile(a, 100-argv.sparsity)).astype(np.float32) for a in dyn_a[0][::argv.subsample].numpy().copy()]))
                del dyn_a
        subsample_a = np.concatenate(subsample_a_list)
        subsample_a_features = np.stack([a[np.triu_indices(a.shape[0], k=1)].copy() for a in subsample_a])
        model = KMeans(n_clusters=argv.num_clusters, random_state=argv.seed).fit(subsample_a_features)
        del subsample_a_features

        print('plot cluster centroids')
        centroid_matrix_list = []
        for centroid in model.cluster_centers_:
            centroid_matrix = np.zeros_like(subsample_a[0])
            centroid_matrix[np.triu_indices_from(centroid_matrix, k=1)] = centroid
            centroid_matrix += centroid_matrix.T
            centroid_matrix_list.append(centroid_matrix)

        fig, ax = plt.subplots(2, 4, figsize=(40,20))

        cluster_cmap = 'jet'
        for cluster, centroid in enumerate(centroid_matrix_list):
            sns.heatmap(data=centroid, cmap=cluster_cmap, vmin=0.0, vmax=1.0, xticklabels=50, yticklabels=50, cbar=True, square=True, ax=ax[cluster//4][cluster%4], cbar_ax=ax[1][3])
            ax[cluster//4][cluster%4].set_title(f'Cluster {cluster+1}')
            ax[cluster//4][cluster%4].set_xticks(list(roi_tick_index.values()))
            ax[cluster//4][cluster%4].set_yticks(list(roi_tick_index.values()))
            ax[cluster//4][cluster%4].set_xticklabels(list(roi_tick_index.keys()), fontsize=10, rotation=45)
            ax[cluster//4][cluster%4].set_yticklabels(list(roi_tick_index.keys()), fontsize=10, rotation=45)
            ax[cluster//4][cluster%4].hlines(list(roi_index.values()), *ax[cluster//4][cluster%4].get_xlim(), colors='white', linestyle='--', linewidth=1.0)
            ax[cluster//4][cluster%4].vlines(list(roi_index.values()), *ax[cluster//4][cluster%4].get_ylim(), colors='white', linestyle='--', linewidth=1.0)

        plt.savefig(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure', 'centroid_matrix.png'))
        plt.close()

        for cluster, centroid in enumerate(centroid_matrix_list):
            fig, ax = plt.subplots(1,2,figsize=(12,10), gridspec_kw={'width_ratios': [25,1]})
            sns.heatmap(data=centroid, cmap=cluster_cmap, vmin=0.0, vmax=1.0, cbar=True, square=True, ax=ax[0], cbar_ax=ax[1])
            ax[0].set_xticks(list(roi_tick_index.values()))
            ax[0].set_yticks(list(roi_tick_index.values()))
            ax[0].set_xticklabels(list(roi_tick_index.keys()), fontsize=10, rotation=45)
            ax[0].set_yticklabels(list(roi_tick_index.keys()), fontsize=10, rotation=45)
            ax[0].hlines(list(roi_index.values()), *ax[0].get_xlim(), colors='white', linestyle='--', linewidth=1.0)
            ax[0].vlines(list(roi_index.values()), *ax[0].get_ylim(), colors='white', linestyle='--', linewidth=1.0)
            plt.suptitle(f'Cluster {cluster+1}')
            plt.savefig(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure', f'centroid_matrix_{cluster+1}.png'))
            plt.close()

            # PLOT DMN
            fig, ax = plt.subplots(1,2,figsize=(12,10), gridspec_kw={'width_ratios': [25,1]})
            centroid_dmn = np.concatenate([centroid[roi_range['L.DMN'][0]:roi_range['L.DMN'][1]], centroid[roi_range['R.DMN'][0]:roi_range['R.DMN'][1]]], axis=0)
            centroid_dmn = np.concatenate([centroid_dmn[:,roi_range['L.DMN'][0]:roi_range['L.DMN'][1]], centroid_dmn[:,roi_range['R.DMN'][0]:roi_range['R.DMN'][1]]], axis=1)
            sns.heatmap(data=centroid_dmn, cmap=cluster_cmap, vmin=0.0, vmax=1.0, cbar=True, square=True, ax=ax[0], cbar_ax=ax[1])
            ax[0].set_xticks(list(dmn_tick_index.values()))
            ax[0].set_yticks(list(dmn_tick_index.values()))
            ax[0].set_xticklabels(list(dmn_tick_index.keys()), fontsize=10, rotation=45)
            ax[0].set_yticklabels(list(dmn_tick_index.keys()), fontsize=10, rotation=45)
            ax[0].hlines(list(dmn_index.values()), *ax[0].get_xlim(), colors='white', linestyle='--', linewidth=1.0)
            ax[0].vlines(list(dmn_index.values()), *ax[0].get_ylim(), colors='white', linestyle='--', linewidth=1.0)
            plt.suptitle(f'Cluster {cluster+1} DMN')
            plt.savefig(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure', f'centroid_matrix_dmn_{cluster+1}.png'))
            plt.close()

            # PLOT SMN
            fig, ax = plt.subplots(1,2,figsize=(12,10), gridspec_kw={'width_ratios': [25,1]})
            centroid_smn = np.concatenate([centroid[roi_range['L.SMN'][0]:roi_range['L.SMN'][1]], centroid[roi_range['R.SMN'][0]:roi_range['R.SMN'][1]]], axis=0)
            centroid_smn = np.concatenate([centroid_smn[:,roi_range['L.SMN'][0]:roi_range['L.SMN'][1]], centroid_smn[:,roi_range['R.SMN'][0]:roi_range['R.SMN'][1]]], axis=1)
            sns.heatmap(data=centroid_smn, cmap=cluster_cmap, vmin=0.0, vmax=1.0, cbar=True, square=True, ax=ax[0], cbar_ax=ax[1])
            plt.suptitle(f'Cluster {cluster+1} SMN')
            plt.savefig(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure', f'centroid_matrix_smn_{cluster+1}.png'))
            plt.close()

        del centroid_matrix_list
        joblib.dump(model, os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'model', f'model_subsample{argv.subsample}.joblib'))

    try:
        significant_clusters = load(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'significant_cluster_count.pkl'))
        chi2dict = load(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'chi2dict.pkl'))
    except:
        print('count significant clusters')
        os.makedirs(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary'), exist_ok=True)
        if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold)
        elif argv.dataset=='ukb-rest': dataset = DatasetUKBRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature)
        else: raise
        significant_clusters = {}
        for layer in range(argv.num_layers):
            significant_clusters[layer] = {}
            for gender in ['F', 'M']:
                significant_clusters[layer][gender] = {}
                for cluster in range(argv.num_clusters):
                    significant_clusters[layer][gender][cluster] = 0
        chi2dict = {}
        for layer in range(argv.num_layers):
            chi2dict[layer] = {'gender':[], 'cluster':[]}

        data_idx = 0
        for k in range(argv.k_fold):
            dataset.set_fold(k, train=False)
            for data in tqdm(dataset, ncols=60):
                dyn_a, _ = bold.process_dynamic_fc(data['timeseries'].unsqueeze(0), argv.window_size, argv.window_stride)
                gender = 'F' if data['label']==0 else 'M'
                for layer in range(argv.num_layers):
                    significant_idx = time_attention_threshold(time_attention[data_idx][layer])
                    if not len(significant_idx)==0:
                        pred = model.predict(np.stack([(a > np.percentile(a, 100-argv.sparsity)).astype(np.float32)[np.triu_indices(a.shape[0], k=1)] for a in dyn_a[0][significant_idx].numpy().copy()]))
                        chi2dict[layer]['gender'].append(data['label']*np.ones_like(pred))
                        chi2dict[layer]['cluster'].append(pred)
                        for c in range(argv.num_clusters):
                            significant_clusters[layer][gender][c] += (pred==c).sum()
                data_idx += 1

        save(significant_clusters, os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'significant_cluster_count.pkl'))
        save(chi2dict, os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'chi2dict.pkl'))

    for layer, dfdict in chi2dict.items():
        dfdict['gender'] = np.concatenate(dfdict['gender'])
        dfdict['cluster'] = np.concatenate(dfdict['cluster'])
        df = pd.DataFrame.from_dict(dfdict)
        expected, observed, statistics = chi2_independence(df, x='gender', y='cluster')
        expected.to_csv(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'chi2_layer{layer}_expected.csv'))
        observed.to_csv(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'chi2_layer{layer}_observed.csv'))
        statistics.to_csv(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'chi2_layer{layer}_statistics.csv'))

    # get rest vs significant ratio
    female_cluster_dict = {}
    male_cluster_dict = {}
    fmratio_cluster_dict = {}
    for layer in range(argv.num_layers):
        female_cluster_dict[f'layer{layer+1}'] = {}
        male_cluster_dict[f'layer{layer+1}'] = {}
        fmratio_cluster_dict[f'layer{layer+1}'] = {}
        for cluster in range(argv.num_clusters):
            female_cluster_dict[f'layer{layer+1}'][f'cluster{cluster+1}'] = None
            male_cluster_dict[f'layer{layer+1}'][f'cluster{cluster+1}'] = None
            fmratio_cluster_dict[f'layer{layer+1}'][f'cluster{cluster+1}'] = None

    for layer, gcv in significant_clusters.items():
        for gender, cv in gcv.items():
            for cluster, v in cv.items():
                rest_ratio = (model.labels_==cluster).sum()/len(model.labels_)
                significant_ratio = v/(sum(cv.values())+1e-9)
                if gender=='F':
                    female_cluster_dict[f'layer{layer+1}'][f'cluster{cluster+1}'] = significant_ratio/rest_ratio
                else:
                    male_cluster_dict[f'layer{layer+1}'][f'cluster{cluster+1}'] = significant_ratio/rest_ratio

    odds_df = {'female_odds': pd.DataFrame.from_dict(female_cluster_dict),
                'male_odds': pd.DataFrame.from_dict(male_cluster_dict)}

    # get female vs male significant ratio
    for layer, cluster_dict in fmratio_cluster_dict.items():
        for cluster in cluster_dict.keys():
            if female_cluster_dict[layer][cluster] > male_cluster_dict[layer][cluster]:
                fmratio_cluster_dict[layer][cluster] = 1.0 * female_cluster_dict[layer][cluster]/(male_cluster_dict[layer][cluster]+1e-9) - 1.0
            else:
                fmratio_cluster_dict[layer][cluster] = -1.0 * (male_cluster_dict[layer][cluster]/(female_cluster_dict[layer][cluster]+1e-9) - 1.0)

    fm_ratio = pd.DataFrame.from_dict(fmratio_cluster_dict)

    fig, ax = plt.subplots(2, 3, figsize=(3*2*argv.num_layers,2*argv.num_clusters), gridspec_kw={'height_ratios':[25,1]})
    for i, (key, odds) in enumerate(odds_df.items()):
        odds.to_csv(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'summary', f'{key}.csv'))
        sns.heatmap(data=odds, xticklabels=True, yticklabels=True, cbar=True, annot=True, center=1.0, vmin=0.2, vmax=1.8, cmap='RdBu_r', ax=ax[0][i], cbar_ax=ax[1][i], cbar_kws={"orientation": "horizontal"})
        ax[0][i].set_title(key)
    sns.heatmap(data=fm_ratio, xticklabels=True, yticklabels=True, cbar=True, annot=True, center=0.0, vmin=-1.8, vmax=1.8, cmap='RdBu_r', ax=ax[0][-1], cbar_ax=ax[1][-1], cbar_kws={"orientation": "horizontal"}, fmt='.1%')
    plt.savefig(os.path.join(analysisdir, 'time_attention', f'{argv.num_clusters}_means_clustering', 'figure', 'odds.png'))
    plt.close()

    # plot dmn-smn figure
    centroid_matrix_list = []
    for centroid in model.cluster_centers_:
        centroid_matrix = np.zeros([400,400])
        centroid_matrix[np.triu_indices_from(centroid_matrix, k=1)] = centroid
        centroid_matrix += centroid_matrix.T
        centroid_matrix_list.append(centroid_matrix)

    odds_df['fmratio_odds'] = odds_df['female_odds'] / odds_df['male_odds']
    cluster_order = odds_df['fmratio_odds'].mean(1).argsort().tolist()[::-1]

    cluster_cmap = 'jet'
    fig, ax = plt.subplot_mosaic([[f'dmn{c}' for c in range(argv.num_clusters)]+['cbar'], [f'smn{c}' for c in range(argv.num_clusters)]+['cbar'], [f'label{c}' for c in range(argv.num_clusters)]+['empty']], figsize=(24,8), gridspec_kw={'width_ratios': [12]*argv.num_clusters+[1], 'height_ratios': [12,12,1]})
    ax['empty'].axis('off')
    for cluster, centroid in enumerate(centroid_matrix_list):
        # PLOT DMN
        centroid_dmn = np.concatenate([centroid[roi_range['L.DMN'][0]:roi_range['L.DMN'][1]], centroid[roi_range['R.DMN'][0]:roi_range['R.DMN'][1]]], axis=0)
        centroid_dmn = np.concatenate([centroid_dmn[:,roi_range['L.DMN'][0]:roi_range['L.DMN'][1]], centroid_dmn[:,roi_range['R.DMN'][0]:roi_range['R.DMN'][1]]], axis=1)
        sns.heatmap(data=centroid_dmn, cmap=cluster_cmap, vmin=0.0, vmax=1.0, xticklabels=False, yticklabels=False, cbar=True if cluster==0 else False, square=True, ax=ax[f'dmn{cluster_order.index(cluster)}'], cbar_ax=ax['cbar'])
        ax[f'dmn{cluster_order.index(cluster)}'].set_title(f'Cluster {cluster+1}', fontsize=24)

        # PLOT SMN
        centroid_smn = np.concatenate([centroid[roi_range['L.SMN'][0]:roi_range['L.SMN'][1]], centroid[roi_range['R.SMN'][0]:roi_range['R.SMN'][1]]], axis=0)
        centroid_smn = np.concatenate([centroid_smn[:,roi_range['L.SMN'][0]:roi_range['L.SMN'][1]], centroid_smn[:,roi_range['R.SMN'][0]:roi_range['R.SMN'][1]]], axis=1)
        sns.heatmap(data=centroid_smn, cmap=cluster_cmap, vmin=0.0, vmax=1.0, xticklabels=False, yticklabels=False, cbar=False, square=True, ax=ax[f'smn{cluster_order.index(cluster)}'])

        # PLOT LABEL
        ax[f'label{cluster_order.index(cluster)}'].text(0.5, 0.5, f'Female: {odds_df["female_odds"].mean(1).to_list()[cluster]:.4f}\nMale: {odds_df["male_odds"].mean(1).to_list()[cluster]:.4f}\nRatio: {odds_df["fmratio_odds"].mean(1).to_list()[cluster]:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=20)
        ax[f'label{cluster_order.index(cluster)}'].axis('off')

    ax['dmn0'].set_ylabel('Default Mode Network')
    ax['smn0'].set_ylabel('Somatomotor Network')

    plt.tight_layout()
    plt.savefig(os.path.join(analysisdir, 'figures', f'{argv.num_clusters}means_clustering_fmratio.png'))
    plt.close()

    # plot roc
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')
    for k in range(argv.k_fold):
        fpr, tpr, thresholds = roc_curve(samples['true'][k], samples['prob'][k][:,1])
        ax.plot(fpr, tpr, label=f'Fold {k+1}', linewidth=3.0)
    ax.plot(np.linspace(0, 1, 100), np.linspace(0,1,100), linestyle='--', linewidth=3.0)
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Receiver operating characteristic curve')
    if argv.k_fold>1: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysisdir, 'figures', 'roc_curve.png'))
    plt.close()

    # plot roi nifti
    node_attention = node_attention.mean((0,2))
    roi = load_img(roidir)
    for layer, layer_node_attention in enumerate(node_attention):
        os.makedirs(os.path.join(analysisdir, 'node_attention', 'nifti', f'layer{layer+1}'), exist_ok=True)
        plot_nifti(layer_node_attention, roi, os.path.join(analysisdir, 'node_attention', 'nifti'), topk=20, prefix=f'layer{layer+1}')

    for layer, layer_node_attention in enumerate(node_attention):
        for i, attention in enumerate(layer_node_attention):
            roidict[i+1][f'node_attention_layer{layer+1}'] = attention
    roidf = pd.DataFrame.from_dict(roidict.values())
    roidf.to_csv(os.path.join(analysisdir, 'node_attention', 'roi_summary.csv'))

    # plot roi summary
    networks = {'Default': 'DMN', 'SalVentAttn': 'SVN', 'Cont': 'CCN', 'DorsAttn': 'DAN', 'Limbic': 'LN', 'SomMot': 'SMN', 'Vis': 'VN'}
    baseline_network_ratio = {}
    for full, shorthand in networks.items():
        baseline_network_ratio[shorthand] = roidf.network.value_counts(full)[full]

    network_ratio = {}
    for v in networks.values():
        network_ratio[v] = []

    sns.set_theme(context='paper', style="dark", font='Freesans', font_scale=2.0, palette='tab10', rc={'text.color': 'white'})
    for layer in range(argv.num_layers):
        os.makedirs(os.path.join(analysisdir, 'node_attention', f'layer{layer+1}'), exist_ok=True)
        network_count = roidf.nlargest(20, f'node_attention_layer{layer+1}').network.value_counts().to_dict()
        for full, shorthand in networks.items():
            if full in network_count.keys():
                network_count[shorthand] = network_count.pop(full)
            else:
                network_count[shorthand] = 0

        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(network_count.values(), startangle=90, counterclock=False)
        plt.legend(labels=network_count.keys(), loc='right', facecolor='black')
        plt.savefig(os.path.join(analysisdir, 'node_attention', f'layer{layer+1}', 'network_pie.png'), facecolor='black')
        plt.close()
        for k, v in network_count.items():
            network_ratio[k].append(v/(sum(network_count.values())+1e-9))

    sns.set_theme(context='paper', style="whitegrid", font='Freesans', font_scale=1.4, palette='tab10')
    fig, ax = plt.subplots(1, len(networks.keys()), figsize=(4*len(networks.keys()),4), sharey='row')
    for n, (network, layer_ratio) in enumerate(network_ratio.items()):
        sns.barplot(x=list(range(argv.num_layers)), y=layer_ratio, ax=ax[n])
        ax[n].set_xticklabels(list(range(1, 1+argv.num_layers)))
        ax[n].set_title(network, fontsize=24.0)
    fig.tight_layout()
    plt.savefig(os.path.join(analysisdir, 'node_attention', 'network_bar.png'))
    plt.close()

    sns.set_theme(context='paper', style="dark", font='Freesans', font_scale=3.0, palette='tab10', rc={'text.color': 'white'})
    for layer in range(argv.num_layers):
        hemisphere_count = roidf.nlargest(20, f'node_attention_layer{layer+1}').hemisphere.value_counts().to_dict()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(hemisphere_count.values(), labels=hemisphere_count.keys(), startangle=90, labeldistance=0.6, counterclock=False)
        plt.savefig(os.path.join(analysisdir, 'node_attention', f'layer{layer+1}', 'hemisphere_pie.png'), facecolor='black')
        plt.close()


def analyze_task(argv):
    task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    analysisdir = os.path.join(argv.targetdir, 'analysis')
    roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(argv.sourcedir, 'roi'))
    roidir = roi['maps']
    roimetadir = os.path.join(argv.sourcedir, 'roi', '7_400_coord.csv')

    if not argv.roi=='schaefer': raise Exception('Schafer atlas is currently supported for analysis')

    os.makedirs(analysisdir, exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'node_attention'), exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'time_attention'), exist_ok=True)
    os.makedirs(os.path.join(analysisdir, 'figures'), exist_ok=True)

    # load data
    node_attention, time_attention, task_block = {}, {}, {}
    for task in task_list:
        node_attention[task] = np.concatenate([np.load(os.path.join(argv.targetdir, 'attention', str(k), 'node_attention', f'{task}.npy')) for k in range(argv.k_fold)])
        time_attention[task] = np.concatenate([np.load(os.path.join(argv.targetdir, 'attention', str(k), 'time_attention', f'{task}.npy')) for k in range(argv.k_fold)])
        task_block_timing = pd.read_csv(os.path.join(argv.sourcedir, 'behavioral', f'hcp_taskrest_{task}.csv'))
        for i, timing in enumerate(task_block_timing):
            task_block_timing[timing] *= (i+1)
        task_block[task] = task_block_timing[12:12+time_attention[task].shape[-1]]

    # summarize roi
    roimeta = pd.read_csv(roimetadir, index_col=0, delimiter=',')
    roimeta = roimeta.loc[1:]
    roidict = {}
    for i, (index, meta) in enumerate(roimeta.iterrows()):
        labels = meta['Label Name'].split('_')[1:-1]
        if len(labels)==2:
            labels.append(labels[-1])
        roidict[i+1] = {'index': index, 'hemisphere': labels[0], 'network': labels[1], 'region': labels[2], 'R': meta['R'], 'A': meta['A'], 'S': meta['S']}

    sns.set_theme(context='paper', style='white', font='Freesans', font_scale=2.0, palette='muted')


    for task in task_list:
        task_roidict = deepcopy(roidict)
        subject_layer_attention = node_attention[task]
        design_matrix = (task_block[task]!=0).to_numpy('int')
        for layer, attention in enumerate(subject_layer_attention.mean(0)):
            for i in range(design_matrix.shape[1]-1):
                contrast = np.zeros(design_matrix.shape[1])
                contrast_type = ['0']*design_matrix.shape[1]
                contrast[i] = 1
                contrast[-1] = -1
                contrast_type[i] = '1'
                contrast_type[-1] = '-1'
                contrast_type = "".join(contrast_type)

                glm_contrast = node_attention_glm(attention, design_matrix, contrast)
                for i, (stat, z_score, p_value) in enumerate(zip(glm_contrast.stat().squeeze(), glm_contrast.z_score().squeeze(), glm_contrast.p_value().squeeze())):
                    task_roidict[i+1][f'layer{layer+1}_{contrast_type}_stat'] = stat
                    task_roidict[i+1][f'layer{layer+1}_{contrast_type}_z_score'] = z_score
                    task_roidict[i+1][f'layer{layer+1}_{contrast_type}_p_value'] = p_value
                    task_roidict[i+1][f'layer{layer+1}_{contrast_type}_significant'] = p_value<0.05
                    task_roidict[i+1][f'layer{layer+1}_{contrast_type}_significant_bonferroni'] = p_value<0.05/(400*argv.num_layers)

        roidf = pd.DataFrame.from_dict(task_roidict.values())
        roidf.to_csv(os.path.join(analysisdir, 'node_attention', f'glm_summary_{task}.csv'))

    for task in task_list:
        fig, ax = plt.subplots(3, 4, figsize=(15,15), gridspec_kw={'width_ratios':[task_block[task].shape[1],25,25,2], 'height_ratios':[25,25,task_block[task].shape[1]]})
        fig.suptitle(f'{task}')
        ax[2][0].axis('off')
        gs = ax[0][3].get_gridspec()
        for _ax in ax[0:3,3]:
            _ax.remove()
        cbar_ax = fig.add_subplot(gs[0:2,3])
        for layer, attention in enumerate(time_attention[task].mean(0)):
            sns.heatmap(ax=ax[layer//2][layer%2+1], data=(attention-attention.min())/(attention.max()-attention.min()), xticklabels=10 if layer//2==1 else False, yticklabels=10 if layer%2==0 else False, cbar=True if layer==0 else False, cbar_ax=cbar_ax)
            ax[layer//2][layer%2+1].set_title(f'Layer {layer+1}')
        sns.heatmap(ax=ax[2][1], data=task_block[task].T, mask=task_block[task].T==0.0, xticklabels=False, yticklabels=True, cbar=False, cmap='Set1')
        sns.heatmap(ax=ax[2][2], data=task_block[task].T, mask=task_block[task].T==0.0, xticklabels=False, yticklabels=False, cbar=False, cmap='Set1')
        sns.heatmap(ax=ax[0][0], data=task_block[task], mask=task_block[task]==0.0, xticklabels=False, yticklabels=False, cbar=False, cmap='Set1')
        sns.heatmap(ax=ax[1][0], data=task_block[task], mask=task_block[task]==0.0, xticklabels=False, yticklabels=False, cbar=False, cmap='Set1')
        for row in ax:
            for col in row:
                for loc in ['top', 'bottom', 'left', 'right']:
                    col.spines[loc].set_visible(True)
        plt.savefig(os.path.join(analysisdir, 'time_attention', f'task_temporal_attention_{task}.png'))
        plt.close()

    roi_index = {'L.VN':0, 'L.SMN':31, 'L.DAN':68, 'L.SVN': 91, 'L.LN': 113, 'L.CCN': 126, 'L.DMN': 148,
                    'R.VN':200, 'R.SMN':230, 'R.DAN':270, 'R.SVN': 293, 'R.LN': 318, 'R.CCN': 331, 'R.DMN': 361}
    roi_tick_index = {'L.VN':15, 'L.SMN':49, 'L.DAN': 79, 'L.SVN': 102, 'L.LN': 119, 'L.CCN': 137, 'L.DMN': 174,
                        'R.VN':215, 'R.SMN':250, 'R.DAN': 281, 'R.SVN': 305, 'R.LN': 324, 'R.CCN': 346, 'R.DMN': 380}

    for task in task_list:
        fig, ax = plt.subplots(2, 4, figsize=(25,15), sharey='row', gridspec_kw={'width_ratios':[task_block[task].shape[1],25,25,2]})
        fig.suptitle(f'{task}')
        gs = ax[0][3].get_gridspec()
        for _ax in ax[:,3]:
            _ax.remove()
        cbar_ax = fig.add_subplot(gs[:,3])
        for layer, attention in enumerate(node_attention[task].mean(0)):
            sns.heatmap(ax=ax[layer//2][layer%2+1], data=(attention-attention.min())/(attention.max()-attention.min()), xticklabels=True, yticklabels=10, cbar=True if layer==0 else False, cbar_ax=cbar_ax)
            ax[layer//2][layer%2+1].set_title(f'Layer {layer+1}')
            ax[layer//2][layer%2+1].tick_params(direction='out', length=5, width=2)
            ax[layer//2][layer%2+1].set_xticks(list(roi_tick_index.values()))
            ax[layer//2][layer%2+1].set_xticklabels(list(roi_tick_index.keys()), rotation=90, fontsize=10)
            ax[layer//2][layer%2+1].vlines(list(roi_index.values()), *ax[layer//2][layer%2+1].get_ylim(), colors='white', linestyle='--', linewidth=1, alpha=1.0)
        sns.heatmap(ax=ax[0][0], data=task_block[task], mask=task_block[task]==0.0, xticklabels=False, yticklabels=False, cbar=False, cmap='Set1')
        sns.heatmap(ax=ax[1][0], data=task_block[task], mask=task_block[task]==0.0, xticklabels=True, yticklabels=False, cbar=False, cmap='Set1')
        for row in ax:
            for col in row:
                for loc in ['top', 'bottom', 'left', 'right']:
                    col.spines[loc].set_visible(True)

        plt.savefig(os.path.join(analysisdir, 'node_attention', f'task_node_attention_{task}.png'))
        plt.close()

    # perform glm
    glm_df = {}
    for task in task_list:
        glm_df[task] = pd.read_csv(os.path.join(analysisdir, 'node_attention', f'glm_summary_{task}.csv'))

    for task in task_list:
        task_roidict = deepcopy(roidict)
        for layer, layer_node_attention in enumerate(node_attention[task].mean(0)):
            for i, attention in enumerate(layer_node_attention.T):
                for c, task_type in enumerate(task_block[task].iloc[:,:-1]):
                    contrast = ['0']*len(task_block[task].T)
                    contrast[c] = '1'
                    contrast[-1] = '-1'
                    contrast = ''.join(contrast)
                    task_roidict[i+1][f'node_attention_{task_type}_layer{layer}'] = attention[task_block[task][task_type].to_numpy().nonzero()[0]].mean(0)
                    task_roidict[i+1][f'glm_zscore_{task_type}_layer{layer}'] = glm_df[task][f'layer{layer+1}_{contrast}_z_score'].values[i]
                    task_roidict[i+1][f'glm_significant_{task_type}_layer{layer}'] = glm_df[task][f'layer{layer+1}_{contrast}_significant_bonferroni'].values[i]
        roidf = pd.DataFrame.from_dict(task_roidict.values())
        roidf.to_csv(os.path.join(analysisdir, 'node_attention', f'roi_summary_{task}.csv'))

    # plot roi summary
    networks = {'Default': 'DMN', 'SalVentAttn': 'SVN', 'Cont': 'CCN', 'DorsAttn': 'DAN', 'Limbic': 'LN', 'SomMot': 'SMN', 'Vis': 'VN'}
    baseline_network_ratio = {}
    for full, shorthand in networks.items():
        baseline_network_ratio[shorthand] = roidf.network.value_counts(full)[full]

    for task in task_list:
        os.makedirs(os.path.join(analysisdir, 'figures', task), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'figures', task, 'network'), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'figures', task, 'network', 'pie'), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'figures', task, 'network', 'bar'), exist_ok=True)
        os.makedirs(os.path.join(analysisdir, 'figures', task, 'hemisphere'), exist_ok=True)
        df = glm_df[task]
        network_task_ratio = {}
        layer_task_ratio = {}
        for task_type in task_block[task].iloc[:,:-1]:
            network_task_ratio[task_type] = {}
            layer_task_ratio[task_type] = {}
            for v in networks.values():
                network_task_ratio[task_type][v] = []
            for layer in range(argv.num_layers):
                layer_task_ratio[task_type][layer] = []
        sns.set_theme(context='paper', style="dark", font='Freesans', font_scale=2.0, palette='tab10', rc={'text.color': 'white'})
        for c, task_type in enumerate(task_block[task].iloc[:,:-1]):
            contrast = ['0']*len(task_block[task].T)
            contrast[c] = '1'
            contrast[-1] = '-1'
            contrast = ''.join(contrast)
            for layer in range(argv.num_layers):
                network_count = df.loc[df[f'layer{layer+1}_{contrast}_significant_bonferroni']==True].network.value_counts().to_dict()
                for full, shorthand in networks.items():
                    if full in network_count.keys():
                        network_count[shorthand] = network_count.pop(full)
                    else:
                        network_count[shorthand] = 0

                fig_pie, ax = plt.subplots(figsize=(6,6))
                ax.pie(network_count.values(), startangle=90, counterclock=False)
                plt.legend(labels=network_count.keys(), loc='right', facecolor='black')
                plt.savefig(os.path.join(analysisdir, 'figures', task, 'network', 'pie', f'layer{layer+1}_{contrast}.png'), facecolor='black')
                plt.close(fig_pie)

                hemisphere_count = df.loc[df[f'layer{layer+1}_{contrast}_significant_bonferroni']==True].hemisphere.value_counts().to_dict()
                fig_hemi, ax = plt.subplots(figsize=(6,6))
                ax.pie(hemisphere_count.values(), labels=hemisphere_count.keys(), startangle=90, labeldistance=0.6, counterclock=False, normalize=True)
                plt.savefig(os.path.join(analysisdir, 'figures', task, 'hemisphere', f'layer{layer+1}_{contrast}.png'), facecolor='black')
                plt.close(fig_hemi)

                for k, v in network_count.items():
                    network_task_ratio[task_type][k].append(v/(sum(network_count.values())+1e-9))
                    layer_task_ratio[task_type][layer].append(v/(sum(network_count.values())+1e-9))

        sns.set_theme(context='paper', style="whitegrid", font='Freesans', font_scale=1.4, palette='tab10')
        fig_bar, ax_bar = plt.subplots(len(task_block[task].T)-1, len(networks.keys()), figsize=(4*len(networks.keys()),4*(len(task_block[task].T)-1)), sharex='col', sharey='row')
        for c, (task_type, network_dict) in enumerate(network_task_ratio.items()):
            for n, (network, layer_ratio) in enumerate(network_dict.items()):
                sns.barplot(x=list(range(argv.num_layers)), y=layer_ratio, ax=ax_bar[c][n] if len(task_block[task].T)>2 else ax_bar[n])
                if len(task_block[task].T)>2: ax_bar[c][n].set_xticklabels(list(range(1, 1+argv.num_layers)))
                else: ax_bar[n].set_xticklabels(list(range(1, 1+argv.num_layers)))
                if c==0:
                    if len(task_block[task].T)>2: ax_bar[c][n].set_title(network, fontsize=24.0)
                    else: ax_bar[n].set_title(network, fontsize=24.0)
                if n==0:
                    if len(task_block[task].T)>2: ax_bar[c][n].set_ylabel(task_type, fontsize=24.0)
                    else: ax_bar[n].set_ylabel(task_type, fontsize=24.0)
        fig_bar.tight_layout()
        plt.savefig(os.path.join(analysisdir, 'figures', task, 'network', 'bar', 'barplot_pernetwork.png'))
        plt.close(fig_bar)

        fig_bar, ax_bar = plt.subplots(len(task_block[task].T)-1, argv.num_layers, figsize=(4*argv.num_layers,4*(len(task_block[task].T)-1)), sharex='col', sharey='row')
        for c, (task_type, layer_dict) in enumerate(layer_task_ratio.items()):
            for l, (layer, network_ratio) in enumerate(layer_dict.items()):
                sns.barplot(x=list(range(len(networks.values()))), y=network_ratio, ax=ax_bar[c][l] if len(task_block[task].T)>2 else ax_bar[l])
                if len(task_block[task].T)>2: ax_bar[c][l].set_xticklabels(networks.values())
                else: ax_bar[l].set_xticklabels(networks.values())
                if c==0:
                    if len(task_block[task].T)>2: ax_bar[c][l].set_title(f'Layer {layer+1}', fontsize=24.0)
                    else: ax_bar[l].set_title(f'Layer {layer+1}', fontsize=24.0)
                if l==0:
                    if len(task_block[task].T)>2: ax_bar[c][l].set_ylabel(task_type, fontsize=24.0)
                    else: ax_bar[l].set_ylabel(task_type, fontsize=24.0)
        fig_bar.tight_layout()
        plt.savefig(os.path.join(analysisdir, 'figures', task, 'network', 'bar', 'barplot_perlayer.png'))
        plt.close(fig_bar)

    # plot roi nifti
    roi = load_img(roidir)
    for task in task_list:
        df = glm_df[task]
        os.makedirs(os.path.join(analysisdir, 'node_attention', task, 'nifti'), exist_ok=True)
        for layer in range(argv.num_layers):
            for c, task_type in enumerate(task_block[task].iloc[:,:-1]):
                contrast = ['0']*len(task_block[task].T)
                contrast[c] = '1'
                contrast[-1] = '-1'
                contrast = ''.join(contrast)
                topk = len(df.loc[df[f'layer{layer+1}_{contrast}_significant_bonferroni']==True])
                node_attention = df[f'layer{layer+1}_{contrast}_z_score'].values
                os.makedirs(os.path.join(analysisdir, 'node_attention', task, 'nifti', f'layer{layer+1}',f'c{contrast}'), exist_ok=True)
                plot_nifti(node_attention, roi, os.path.join(analysisdir, 'node_attention', task, 'nifti'), topk=topk, prefix=f'layer{layer+1}/c{contrast}')


def analyze(argv):
    assert argv.roi == 'schaefer'

    # set seed
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(argv.seed)

    if 'rest' in argv.dataset: analyze_rest(argv)
    elif 'task' in argv.dataset: analyze_task(argv)
    else: raise


if __name__ == '__main__':
    argv = option.parse()
    analyze(argv)
    exit(0)
