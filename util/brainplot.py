# mricrogl>=1.2 python script
# run example:
# >> MRIcroGL util/brainplot.py

import os
import gl
import cv2
import numpy as np
import pandas as pd

EXP_DIR='result/stagin_experiment'
ROIMETA_DIR = 'data/roi/7_400_coord.csv'
NETWORK_LIST = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

VMIN = 0.55
VMAX = 1.0
TASK = ''
# VMIN = 4.0
# VMAX = 13.5
# TASK = 'WM'


def sort_rois(roidir, roimetadir):
    roi_dict = {}
    for network in NETWORK_LIST:
        roi_dict[network] = {'LH': [], 'RH': []}

    roimeta = pd.read_csv(roimetadir, index_col=0, delimiter=',')
    roimeta = roimeta.loc[1:]

    roi_list = os.listdir(roidir)
    for roi in roi_list:
        index = int(roi.split('.')[0].split('_')[1][3:])

        labels = roimeta.loc[index]['Label Name'].split('_')[1:3]
        roi_dict[labels[1]][labels[0]].append(os.path.join(os.path.abspath(roidir), roi))

    return roi_dict


def initialize(atlas='mni152'):
    gl.resetdefaults()
    gl.loadimage(atlas)
    gl.overlayloadsmooth(True)
    gl.opacity(0, 50)
    gl.colorbarposition(0)


def visualize_axial(roi_dict, savedir, min=VMIN, max=VMAX, colorname='4hot'):
    gl.viewaxial(1)
    for network in NETWORK_LIST:
        network_roi = roi_dict[network]['LH'] + roi_dict[network]['RH']
        initialize()
        for i, roi in enumerate(network_roi):
            gl.overlayload(roi)
            gl.wait(100)
            gl.minmax(i+1, min, max)
            gl.wait(10)
            gl.colorname(i+1, colorname)
            gl.wait(10)
        gl.viewaxial(1)
        gl.savebmp(os.path.join(savedir, f'{network}_axial.png'))
        gl.overlaycloseall()


def visualize_sagittal(roi_dict, savedir, min=VMIN, max=VMAX, colorname='4hot'):
    for hemisphere in ['LH', 'RH']:
        for network in NETWORK_LIST:
            initialize()
            if hemisphere =='LH': gl.clipazimuthelevation(0.49, 90, 0)
            elif hemisphere =='RH': gl.clipazimuthelevation(0.49, 270, 0)
            else: raise
            for i, roi in enumerate(roi_dict[network][hemisphere]):
                gl.overlayload(roi)
                gl.wait(100)
                gl.minmax(i+1, min, max)
                gl.wait(10)
                gl.colorname(i+1, colorname)
                gl.wait(10)

            gl.viewsagittal(1)
            gl.savebmp(os.path.join(savedir, f'{network}_{hemisphere}_sagittal_lt.png'))
            gl.viewsagittal(0)
            gl.savebmp(os.path.join(savedir, f'{network}_{hemisphere}_sagittal_rt.png'))
            gl.overlaycloseall()


def visualize_colorbar(savedir, min=VMIN, max=VMAX, colorname='4hot'):
    gl.resetdefaults()
    gl.minmax(0, min, max)
    gl.colorname(0, colorname)
    gl.opacity(0, 0)
    gl.colorbarposition(1)
    gl.savebmp(os.path.join(savedir, 'colorbar.png'))


def plot_figure(sourcedir, targetdir=None):
    if targetdir is None: targetdir=sourcedir
    network_figures = {}
    for network in NETWORK_LIST:
        axial = cv2.resize(cv2.imread(os.path.join(sourcedir, f'{network}_axial.png'))[150:-150, 300:-300], dsize=(0,0), fx=1.6, fy=1.6)
        sagittal_lh_lt = cv2.imread(os.path.join(sourcedir, f'{network}_LH_sagittal_lt.png'))[200:-200, 200:-200]
        sagittal_rh_lt = cv2.imread(os.path.join(sourcedir, f'{network}_RH_sagittal_lt.png'))[200:-200, 200:-200]
        sagittal_lh_rt = cv2.imread(os.path.join(sourcedir, f'{network}_LH_sagittal_rt.png'))[200:-200, 200:-200]
        sagittal_rh_rt = cv2.imread(os.path.join(sourcedir, f'{network}_RH_sagittal_rt.png'))[200:-200, 200:-200]
        sagittal_lt = np.concatenate([sagittal_lh_lt, sagittal_rh_lt])
        sagittal_rt = np.concatenate([sagittal_rh_rt, sagittal_lh_rt])
        axial = np.concatenate([axial, np.zeros_like(axial)[:len(sagittal_lt)-len(axial)]])
        try:
            network_figure = np.concatenate([sagittal_lt, axial, sagittal_rt], axis=1)
        except:
            print(network)
            print(sagittal_lt.shape)
            print(sagittal_rt.shape)
            print(axial.shape)
        cv2.imwrite(os.path.join(sourcedir, f'{network}_full.png'), network_figure)
        network_figures[network] = np.concatenate([np.zeros_like(network_figure)[:,:50], network_figure, np.zeros_like(network_figure)[:,:50]], axis=1)
    full_figure = np.concatenate([np.concatenate([network_figures['Default'], network_figures['SalVentAttn'], network_figures['Cont'], network_figures['DorsAttn']], axis=1), np.concatenate([network_figures['Limbic'], network_figures['SomMot'], network_figures['Vis'], np.zeros_like(network_figures['Vis'])], axis=1)])
    cv2.imwrite(os.path.join(targetdir, 'network.png'), full_figure)


if __name__=='__main__':
    # plot node attention
    if TASK:
        for layer in os.listdir(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'nifti')):
            for contrast in os.listdir(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'nifti', layer)):
                os.makedirs(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'image', layer, contrast), exist_ok=True)
                os.makedirs(os.path.join(EXP_DIR, 'analysis', 'figures', TASK, layer, contrast), exist_ok=True)
                roi_dict_node_attention = sort_rois(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'nifti', layer, contrast), ROIMETA_DIR)

                visualize_axial(roi_dict_node_attention, os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'image', layer, contrast))
                visualize_sagittal(roi_dict_node_attention, os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'image', layer, contrast))
                plot_figure(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'image', layer, contrast), os.path.join(EXP_DIR, 'analysis', 'figures', TASK, layer, contrast))
        visualize_colorbar(os.path.join(EXP_DIR, 'analysis', 'node_attention', TASK, 'image'))

    else:
        for layer in os.listdir(os.path.join(EXP_DIR, 'analysis', 'node_attention', 'nifti')):
            os.makedirs(os.path.join(EXP_DIR, 'analysis', 'node_attention', 'image', layer), exist_ok=True)
            os.makedirs(os.path.join(EXP_DIR, 'analysis', 'figures', layer), exist_ok=True)
            roi_dict_node_attention = sort_rois(os.path.join(EXP_DIR, 'analysis', 'node_attention', 'nifti', layer), ROIMETA_DIR)

            visualize_axial(roi_dict_node_attention, os.path.join(EXP_DIR, 'analysis', 'node_attention', 'image', layer))
            visualize_sagittal(roi_dict_node_attention, os.path.join(EXP_DIR, 'analysis', 'node_attention', 'image', layer))
            plot_figure(os.path.join(EXP_DIR, 'analysis', 'node_attention', 'image', layer), os.path.join(EXP_DIR, 'analysis', 'figures', layer))
        visualize_colorbar(os.path.join(EXP_DIR, 'analysis', 'node_attention', 'image'))

    exit(0)
