#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.


import sys, os, pdb
from os.path import join as ospj
from os.path import basename as ospb
from os.path import dirname as ospd
import numpy as np
import torch
from collections import *
from itertools import *
import pandas as pd
import pickle, json, csv
from tqdm import tqdm
from pandas.core.common import flatten
import ipdb
import pickle

# Custom
import preprocess
import dutils
import viz
import argparse

"""
Script to load BABEL segments with NTU skeleton format and pre-process.
"""


def ntu_style_preprocessing(b_dset_path):
    '''
    '''
    pdb.set_trace()
    print('Load BABEL v1.0 dataset subset', b_dset_path)
    b_dset = dutils.read_pkl(b_dset_path)
    # Get unnormalized 5-sec. samples
    X = np.array(b_dset['X'])
    print('X (old) = ', np.shape(X))  # N, T, V, C

    # Prep. data for normalization
    X = X.transpose(0, 3, 1, 2)  # N, C, T, V
    X = X[:, :, :, :, np.newaxis]  # N, C, T, V, M
    print('Shape of prepped X: ', X.shape)

    # Normalize (pre-process) in NTU RGBD-style
    ntu_sk_spine_bone = np.array([0, 1])
    ntu_sk_shoulder_bone = np.array([8, 4])
    X, l_m_sk = preprocess.pre_normalization(X, zaxis=ntu_sk_spine_bone,
                                                    xaxis=ntu_sk_shoulder_bone)
    print('Shape of normalized X: ', X.shape)
    print('Skipping {0} samples because "skeleton is missing"'.format(len(l_m_sk)))
    print('Skipped idxs = ', l_m_sk)

    # Dataset w/ processed seg. chunks. (Skip samples w/ missing skeletons)
    b_AR_dset = {k: np.delete(b_dset[k], l_m_sk) for k in b_dset if k!='X'}
    b_AR_dset['X'] = np.delete(X, l_m_sk, axis=0)
    print('Shape of dataset = ', b_AR_dset['X'].shape)

    fp = b_dset_path.replace('samples', 'ntu_sk_ntu-style_preprocessed' )
    # fp = '../data/babel_v1.0/babel_v1.0_ntu_sk_ntu-style_preprocessed.pkl'
    # dutils.write_pkl(b_AR_dset, fp)
    with open(fp, 'wb') as of:
        pickle.dump(b_AR_dset, of, protocol=4)

def get_act_idx(y, act2idx, n_classes):
    '''
    '''
    if y in act2idx:
        return act2idx[y]
    else:
        return n_classes

def store_splits_subsets(n_classes, spl, plus_extra = True, w_folder = '../data/babel_v1.0/', sk_type = 'nturgbd'):
    '''
    '''
    # Get splits
    splits = dutils.read_json('../data/amass_splits.json')
    sid2split = {int(ospb(u).replace('.mp4', '')): spl for spl in splits \
                                                        for u in splits[spl] }

    # In labels, act. cat. --> idx
    act2idx_150 = dutils.read_json('../data/action_label_2_idx.json')
    act2idx = {k: act2idx_150[k] for k in act2idx_150 if act2idx_150[k] < n_classes}
    print('{0} actions in label set: {1}'.format(len(act2idx), act2idx))

    if sk_type == 'nturgbd':
        if plus_extra :
            fp = ospj(w_folder, 'babel_v1.0_'+spl+'_extra_ntu_sk_ntu-style_preprocessed.pkl')
        else:
            fp = ospj(w_folder, 'babel_v1.0_'+spl+'_ntu_sk_ntu-style_preprocessed.pkl')
    else:
        if plus_extra :
            fp = ospj(w_folder, 'babel_v1.0_'+spl+'_extra_samples.pkl')
        else:
            fp = ospj(w_folder, 'babel_v1.0_'+spl+'_samples.pkl')
    # Get full dataset
    b_AR_dset = dutils.read_pkl(fp)

    # Store idxs of samples to include in learning
    split_idxs = defaultdict(list)
    for i, y1 in enumerate(b_AR_dset['Y1']):

        # Check if action category in list of classes
        if y1 not in act2idx:
            continue

        sid = b_AR_dset['sid'][i]
        split_idxs[sid2split[sid]].append(i)  # Include idx in dataset

    # Save features that'll be loaded by dataloader
    ar_idxs = np.array(split_idxs[spl])
    X = []
    for idx in split_idxs[spl]:
        X.append(b_AR_dset['X'][idx]) # append to other list instead of array conversion (save RAM)
    X = np.array(X)

    if sk_type == 'nturgbd':
        if plus_extra:
            fn = ospj(w_folder, f'{spl}_extra_ntu_sk_{n_classes}.npy')
        else:
            fn = ospj(w_folder, f'{spl}_ntu_sk_{n_classes}.npy')
    else:
        if plus_extra:
            fn = ospj(w_folder, f'{spl}_extra_' + sk_type + f'_{n_classes}.npy')
        else:
            fn = ospj(w_folder, f'{spl}_' + sk_type + f'_{n_classes}.npy')
    np.save(fn, X)

    # labels
    labels = {k: np.array(b_AR_dset[k], dtype=object)[ar_idxs] for k in b_AR_dset if k!='X'}

    # Create, save label data structure that'll be loaded by dataloader
    label_idxs = defaultdict(list)
    for i, y1 in enumerate(labels['Y1']):
        label_idxs['Y1'].append(act2idx[y1]) # TODO : check how "transition" class if handled
        # yk
        yk = [get_act_idx(y, act2idx, n_classes) for y in labels['Yk'][i]]
        label_idxs['Yk'].append(yk)
        # yov
        yov_o = labels['Yov'][i]
        yov = {get_act_idx(y, act2idx, n_classes): yov_o[y] for y in yov_o}
        label_idxs['Yov'].append(yov)
        #
        label_idxs['seg_id'].append(labels['seg_id'][i])
        label_idxs['sid'].append(labels['sid'][i])
        label_idxs['chunk_n'].append(labels['chunk_n'][i])
        label_idxs['anntr_id'].append(labels['anntr_id'][i])

    if plus_extra:
        wr_f = ospj(w_folder,  f'{spl}_extra_label_{n_classes}.pkl')
    else:
        wr_f = ospj(w_folder, f'{spl}_label_{n_classes}.pkl')
    dutils.write_pkl(\
        (label_idxs['seg_id'], (label_idxs['Y1'], label_idxs['sid'],
                                label_idxs['chunk_n'], label_idxs['anntr_id'])), \
                                wr_f)


class Babel_AR:
    '''Object containing data, methods for Action Recognition.

    Task
    -----
    Given: x (Segment from Babel)
    Predict: \hat{p}(x) (Distribution over action categories)

    GT
    ---
    How to compute GT for a given segment?
    - yk: All action categories that are labeled for the entirety of segment
    - y1: One of yk
    - yov: Any y that belongs to part of a segment is considered to be GT.
           Fraction of segment covered by an action: {'walk': 1.0, 'wave': 0.5}

    '''
    def __init__(self, dataset, dense=True, seq_dense_ann_type={}, sk_type='nturgbd', jpos_p='../data/', split_files=False):
        '''Dataset with (samples, different GTs)
        '''
        # Load dataset
        self.babel = dataset
        self.dense = dense
        self.seq_dense_ann_type = seq_dense_ann_type
        self.sk_type = sk_type
        self.jpos_p = jpos_p

        # Get frame-rate for each seq. in AMASS
        f_p = '../data/featp_2_fps.json'
        self.ft_p_2_fps = dutils.read_json(f_p)

        # Dataset w/ keys = {'X', 'Y1', 'Yk', 'Yov', 'seg_id',  'sid',
        # 'seg_dur'}
        if split_files:
            self.d = []
            for ann in tqdm(self.babel):
                self._update_dataset_for_split(ann)
        else:
            self.d = defaultdict(list)
            for ann in tqdm(self.babel):
                self._update_dataset(ann)

    def _subsample_to_30fps(self, orig_ft, orig_fps):
        '''Get features at 30fps frame-rate
        Args:
            orig_ft <array> (T, 25*3): Feats. @ `orig_fps` frame-rate
            orig_fps <float>: Frame-rate in original (ft) seq.
        Return:
            ft <array> (T', 25*3): Feats. @ 30fps
        '''
        T, n_j, _ = orig_ft.shape
        out_fps = 30.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps)%int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T/int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration*int(orig_fps)
                t_idxs += [x+ last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T ]
        else:
            t_idxs = np.arange(0, T, orig_fps/out_fps, dtype=int)

        ft = orig_ft[t_idxs, :, :]
        return ft

    def _viz_x(self, ft, fn='test_sample'):
        '''Wraper to Viz. the given sample (w/ NTU RGBD skeleton)'''
        viz.viz_seq(seq=ft, folder_p=f'test_viz/{fn}', sk_type=self.sk_type,
                                                                    debug=True)
        return None

    def _load_seq_feats(self, ft_p, sk_type):
        '''Given path to joint position features, return them in 30fps'''
        # Identify appropriate feature directory path on disk
        if 'smpl_wo_hands' == sk_type:  # SMPL w/o hands (T, 22*3)
            jpos_p = ospj(self.jpos_p, 'joint_pos')
        if 'nturgbd' == sk_type or 'h36m' == sk_type:  # NTU (T, 219)
            jpos_p = ospj(self.jpos_p, 'babel_joint_pos')
        if 'smplh' == sk_type: # SMPLH (T, 52*3) (joint angles)
            jpos_p = ospj(self.jpos_p, 'smplh')
        if 'smplx' == sk_type: # SMPLH (T, 54*3) (joint angles)
            jpos_p = ospj(self.jpos_p, 'smplx')
        
        if not sk_type == 'smplh':
            # Get the correct dataset folder name
            ddir_n = ospb(ospd(ospd(ft_p)))
            ddir_map = {'BioMotionLab_NTroje': 'BMLrub', 'DFaust_67': 'DFaust'}
            ddir_n = ddir_map[ddir_n] if ddir_n in ddir_map else ddir_n
            # Get the subject folder name
            sub_fol_n = ospb(ospd(ft_p))

            # Sanity check
            fft_p = ospj(jpos_p, ddir_n, sub_fol_n, ospb(ft_p))
            assert os.path.exists(fft_p), fft_p

            # Load seq. fts.
            ft = np.load(fft_p)['joint_pos']
        else:
            fft_p = ospj(jpos_p, ft_p)
            assert os.path.exists(fft_p), fft_p

            ft = np.load(fft_p)['poses']

        T, ft_sz = ft.shape

        if sk_type == "nturgbd":
            # Get NTU skeleton joints
            ntu_js = dutils.smpl_to(model_type='smplh', out_format=sk_type)
            ft = ft.reshape(T, -1, 3)
            ft = ft[:, ntu_js, :]
        else:
            ft = ft.reshape(T, -1, 3)

        # Sub-sample to 30fps
        orig_fps = self.ft_p_2_fps[ft_p]
        ft = self._subsample_to_30fps(ft, orig_fps)
        # print(f'Feat. shape = {ft.shape}, fps = {orig_fps}')
        # if orig_fps != 30.0:
        #   self._viz_x(ft)
        return ft
    
    def _load_seq_poses(self, ft_p, sk_type):
        if 'smplh' == sk_type: # SMPLH (T, 52*3) (joint angles)
            jpos_p = ospj(self.jpos_p, 'smplh')
            ft_p_tmp = ft_p
        if 'smplx' == sk_type: # SMPLH (T, 52*3) (joint angles)
            jpos_p = ospj(self.jpos_p, 'smplx')
            ft_p_tmp = ft_p.replace('_poses.npz', '_stageii.npz')
            ft_p_tmp = ft_p_tmp.replace(' ', '_')

        fft_p = ospj(jpos_p, ft_p_tmp)
        # assert os.path.exists(fft_p), fft_p
        if not os.path.exists(fft_p):
            print(fft_p, " does not exist")
            return None, None, None

        ft = np.load(fft_p)
        poses = ft['poses']
        trans = ft['trans']

        poses = self._orig_sample_to_30fps(poses, ft_p)
        trans = self._orig_sample_to_30fps(trans, ft_p)
        
        return poses, trans, ft['betas']

        # ft = ft.reshape(T, -1, 3)

        # T, ft_sz = ft.shape

        # orig_fps = self.ft_p_2_fps[ft_p]
        # ft = self._subsample_to_30fps(ft, orig_fps)

    def _orig_sample_to_30fps(self, ft, ft_p):
        T, ft_sz = ft.shape

        ft = ft.reshape(T, -1, 3)

        orig_fps = self.ft_p_2_fps[ft_p]
        ft = self._subsample_to_30fps(ft, orig_fps)
        return ft

    def _get_per_f_labels(self, ann, ann_type, seq_dur):
        ''' '''
        # Per-frame labels: {0: ['walk'], 1: ['walk', 'wave'], ... T: ['stand']}
        yf = defaultdict(list)
        T = int(30.0*seq_dur)
        for n_f in range(T):
            cur_t = float(n_f/30.0)
            for seg in ann['labels']:

                if seg['act_cat'] is None:
                    continue

                if 'seq_ann' == ann_type:
                    seg['start_t'] = 0.0
                    seg['end_t'] = seq_dur

                if cur_t >= float(seg['start_t']) and cur_t < float(seg['end_t']):
                    yf[n_f] += seg['act_cat']
        return yf

    def _compute_dur_samples(self, ann, ann_type, seq_ft, seq_dur, dur=5.0, seq_trans=None, do_not_equalize=False):
        '''Return each GT action, corresponding to the fraction of the
        segment that it overlaps with.
        There are 2 conditions that we need to handle:
            1. Multiple action categories in 'act_cat'
            2. Simultaneous (overlapping action segments).

        Example Input:
        Seq. => frames [0,      1,      2,      3,      4,      5]
        GT acts. =>  [[2,3],  [2,3],    [2],    [0],    [0,1],  [0,1]]

        Segs, GT:
        1. seg_x = seq[0: 3], y1 = 2, yall = {2: 1.0, 3: 0.66}
        2. seg_x = seq[0: 2], y1 = 3, yall = {2: 1.0, 3: 1.0}
        3. seg_x = seq[3: ],  y1 = 0, yall = {0: 1.0, 1: 0.66}
        4. seg_x = seq[4: ],  y1 = 1, yall = {0: 1.0, 1: 1.0}

        - Note that we should do the above for each chunk in a segment,
        each of duration = <dur> seconds.

        Return:
        [ { 'x': [st_t, end_t],
            'y1': <act. cat.>,
            'yall': { <act. cat.>: <fraction in seg>, ...}},
          { ... }, ...
        ]
        '''
        #
        yf = self._get_per_f_labels(ann, ann_type, seq_dur)

        # Compute, store all samples for each segment
        seq_samples = []
        for seg in ann['labels']:

            # If no labeled act. cats. for current seg., skip it
            if seg['act_cat'] is None or 0 == len(seg['act_cat']):
                continue

            # Handle stage 1 missing durs.
            if 'seq_ann' == ann_type:
                seg['start_t'] = 0.0
                seg['end_t'] = seq_dur

            # Get segment feats.
            seg_st_f, seg_end_f = int(30.0*seg['start_t']), int(30.0*seg['end_t'])
            seg_x = seq_ft[seg_st_f: seg_end_f, :, :]
            if seq_trans is not None:
                seg_trans = seq_trans[seg_st_f: seg_end_f, :, :]


            # Split segment into <n_chunks> <dur>-second chunks
            n_f_pc = 30.0 * dur
            n_chunks = int(np.ceil(seg_x.shape[0]/n_f_pc))
            for n_ch in range(n_chunks):

                # Single <dur>-sec. chunk in segment
                ch_st_f = int(n_f_pc * n_ch)
                ch_end_f = int(min(ch_st_f + n_f_pc, seg_x.shape[0]))
                x = seg_x[ch_st_f: ch_end_f, :, :]

                x_T, nj, xyz = x.shape
                if not do_not_equalize:
                    # Handle case where chunk_T < n_f_pc
                    x_ch = np.concatenate((x, np.zeros((int(n_f_pc)- x_T,  nj, xyz))), axis=0)
                else:
                    x_ch = x


                if seq_trans is not None:
                    trans = seg_trans[ch_st_f: ch_end_f, :, :]
                    if not do_not_equalize:
                        trans_T, nj_T, xyz_T = trans.shape
                        trans_ch = np.concatenate((trans, np.zeros((int(n_f_pc)- trans_T,  nj_T, xyz_T))), axis=0)
                    else:   
                        trans_ch = trans

                # Labels for this chunk
                yov = Counter(flatten([yf[seg_st_f + n_f] for n_f in range(ch_st_f, ch_end_f)]))

                # Sanity check -- is segment smaller than 1 frame?
                if seg['act_cat'][0] not in yov:
                    # print('Skipping seg:', seg)
                    # print(f'Chunk # {n_ch}, Yov: ', yov)
                    continue

                yov = {k: round(yov[k]/x_T, 3) for k in yov}

                # For each act_cat in segment, create a separate sample
                for cat in seg['act_cat']:
                    # Add to samples GTs
                    seq_sample = {'seg_id': seg['seg_id'],
                                        'chunk_n': n_ch,
                                        'chunk_dur': round(x_T/n_f_pc, 3),
                                        'x': x_ch,
                                        'y1': cat,
                                        'yk': seg['act_cat'],
                                        'yov': yov,
                                        'anntr_id': ann['anntr_id']
                                       }
                    if seq_trans is not None:
                        seq_sample['trans'] = trans_ch

                    seq_samples.append(seq_sample)

        return seq_samples

    def _sample_at_seg_chunk_level(self, ann, seq_samples):
        # Samples at segment-chunk-level
        for i, sample in enumerate(seq_samples):

            self.d['sid'].append(ann['babel_sid'])  # Seq. info
            self.d['seg_id'].append(sample['seg_id'])  # Seg. info
            self.d['chunk_n'].append(sample['chunk_n'])  # Seg. chunk info
            self.d['anntr_id'].append(sample['anntr_id']) # Annotator id (useful in rebuttal exp.)
            self.d['chunk_dur'].append(sample['chunk_dur'])  # Seg. chunk info
            self.d['X'].append(sample['x'])  # Seg. chunk feats.
            self.d['Y1'].append(sample['y1'])  # 1 out of k GT act. cats.
            self.d['Yk'].append(sample['yk'])  # List of k GT act. cats.
            # <dict>: fractions of overlapping act. cats.
            self.d['Yov'].append(sample['yov'])
        return
    

    def _update_dataset(self, ann):
        '''Return one sample (one segment) = (X, Y1, Yall)'''

        # Get feats. for seq.
        if 'smpl' not in self.sk_type:
            seq_ft = self._load_seq_feats(ann['feat_p'], self.sk_type)

        # To keep track of type of annotation for loading 'extra'
        # Compute all GT labels for this seq.
        seq_samples = None
        if self.dense:
            if ann['frame_ann'] is not None:
                ann_ar = ann['frame_ann']
                self.seq_dense_ann_type[ann['babel_sid']] = 'frame_ann'
                seq_samples = self._compute_dur_samples(ann_ar, 'frame_ann', seq_ft, ann['dur'])
            else:
                ann_ar = ann['seq_ann']
                self.seq_dense_ann_type[ann['babel_sid']] = 'seq_ann'
                seq_samples = self._compute_dur_samples(ann_ar, 'seq_ann', seq_ft, ann['dur'])
            self._sample_at_seg_chunk_level(ann, seq_samples)
        else:
            # check if extra exists
            if 'frame_anns' in ann.keys() or 'seq_anns' in ann.keys():
                ann_type = None
                if ann['babel_sid'] in self.seq_dense_ann_type:
                    ann_type = self.seq_dense_ann_type[ann['babel_sid']]
                else:
                    if ann['frame_anns'] is not None:
                        ann_type = 'frame_ann'
                    elif ann['seq_anns'] is not None:
                        ann_type = 'seq_ann'
                    else:
                        ipdb.set_trace()
                    self.seq_dense_ann_type['babel_sid'] = ann_type
                ann_ar = None
                if ann_type == 'frame_ann':
                    if ann['frame_anns'] is not None:
                        ann_ar = ann['frame_anns']
                elif ann_type == 'seq_ann':
                    if ann['seq_anns'] is not None:
                        ann_ar = ann['seq_anns']
                else:
                    ipdb.set_trace()
                if ann_ar:
                    for an in ann_ar:
                        seq_samples = self._compute_dur_samples(an, ann_type, \
                                                                seq_ft, ann['dur'])
                        self._sample_at_seg_chunk_level(ann, seq_samples)
            else:
                print('Unexpected format for extra!')
        return
    
    def _sample_at_seg_chunk_level_for_split(self, ann, seq_samples, betas):
        for sample in seq_samples:
            # print('pose sample: ', sample['x'].shape)
            seg = {}
            seg['sid'] = ann['babel_sid']
            seg['seg_id'] = sample['seg_id']
            seg['Y1'] = sample['y1']
            seg['poses'] = sample['x'].reshape((sample['x'].shape[0], -1))
            seg['trans'] = sample['trans'].reshape((sample['trans'].shape[0], -1))
            seg['betas'] = betas
            seg['mocap_frame_rate'] = 30.0
            seg['gender'] = 'neutral'
            self.d.append(seg)

    def _update_dataset_for_split(self, ann):
        seq_poses, seq_trans, betas = self._load_seq_poses(ann['feat_p'], self.sk_type)
        if seq_poses is None:
            return

        seq_samples = None
        if ann['frame_ann'] is not None:
            ann_ar = ann['frame_ann']
            self.seq_dense_ann_type[ann['babel_sid']] = 'frame_ann'
            seq_samples = self._compute_dur_samples(ann_ar, 'frame_ann', seq_poses, ann['dur'], seq_trans=seq_trans, do_not_equalize=True)
        else:
            ann_ar = ann['seq_ann']
            self.seq_dense_ann_type[ann['babel_sid']] = 'seq_ann'
            seq_samples = self._compute_dur_samples(ann_ar, 'seq_ann', seq_poses, ann['dur'], seq_trans=seq_trans, do_not_equalize=True)
        self._sample_at_seg_chunk_level_for_split(ann, seq_samples, betas)


def parse_args():
    parser = argparse.ArgumentParser(description="Create datasets from splits and joint locations")
    parser.add_argument('--splits_folder', type=str, default='../data/babel_v1.0_release/', help="Path to the folder containing splits")
    parser.add_argument('--joints_folder', type=str, default='../data', help="Path to the folder containing joint locations")
    parser.add_argument('--dst_folder', type=str, default='../data/babel_v1.0/', help="Path to the destination folder")
    parser.add_argument('--create_extra', action="store_true", default=False, help="Set if extra data should be processed")
    parser.add_argument('--in_several_files', action="store_true", default=False, help="Set if one file per action should be extracted")
    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()

    #  Create dataset
    # --------------------------
    # d_folder = '../data/babel_v1.0_release/'
    d_folder = args.splits_folder
    w_folder = args.dst_folder
    # w_folder = '../data/babel_v1.0/'
    # sk_type = 'smplh'
    sk_type = 'smplx'
    for spl in ['train', 'val']:

        # Load Dense BABEL
        data = dutils.read_json(ospj(d_folder, f'{spl}.json'))
        dataset = [data[sid] for sid in data]
        if args.in_several_files:
            dense_babel = Babel_AR(dataset, dense=True, sk_type=sk_type, jpos_p=args.joints_folder, split_files=True)
            folder_spl = ospj(w_folder, spl)
            os.makedirs(folder_spl, exist_ok=True)
            print("export files in split folders")
            for d in tqdm(dense_babel.d):
                f_out_path = ospj(folder_spl, str(d['sid']) + '_' + str(d['seg_id']) + '_' + str(d['Y1']).replace('/','-') + '.npz')
                np.savez(f_out_path, **d)
        else:
            dense_babel = Babel_AR(dataset, dense=True, sk_type=sk_type, jpos_p=args.joints_folder, split_files=False)
            # Store Dense BABEL
            d_filename = ospj(w_folder, 'babel_v1.0_'+ spl + '_samples.pkl')
            dutils.write_pkl(dense_babel.d, d_filename)

            if args.create_extra:
                # Load Extra BABEL
                data = dutils.read_json(ospj(d_folder, f'extra_{spl}.json'))
                dataset = [data[sid] for sid in data]
                extra_babel = Babel_AR(dataset, dense=False,
                                    seq_dense_ann_type=dense_babel.seq_dense_ann_type,
                                    jpos_p=args.joints_folder)
                # Store Dense + Extra
                de = {}
                for k in dense_babel.d.keys():
                    de[k] = dense_babel.d[k] + extra_babel.d[k]
                ex_filename = w_folder + 'babel_v1.0_' + spl + '_extra_samples.pkl'
                dutils.write_pkl(de, ex_filename)

            if sk_type == 'nturgbd':
                #  Pre-process, Store data in dataset
                print('NTU-style preprocessing')
                babel_dataset_AR = ntu_style_preprocessing(d_filename)
                if args.create_extra:
                    babel_dataset_AR = ntu_style_preprocessing(ex_filename)

            for C in (120, 60):
                store_splits_subsets(n_classes=C, spl=spl, plus_extra=False, sk_type=sk_type, w_folder=args.dst_folder)
                label_fp = ospj(w_folder, f'{spl}_label_{C}.pkl')
                dutils.store_counts(label_fp)

            if args.create_extra:
                ex = '_extra'
                for C in (120, 60):
                    store_splits_subsets(n_classes=C, spl=spl, plus_extra=True, sk_type=sk_type, w_folder=args.dst_folder)
                    label_fp = ospj(w_folder, f'{spl}{ex}_label_{C}.pkl')
                    dutils.store_counts(label_fp)