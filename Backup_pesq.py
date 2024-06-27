
import torch
import torch.nn as nn
import librosa
import pickle
import json
import os
import h5py
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from scipy import signal
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
import scipy.linalg as linalg
from config_merge import feat_type, fft_num, win_size, win_shift
import math

EPSILON = 1e-10


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    return loss


def com_mag_mse_loss(esti, label, frame_list):
    utt_num = esti.size()[0]
    with torch.no_grad():
        esti_mag, esti_phase = torch.norm(esti, dim=1), torch.atan2(esti[:, -1, :, :], esti[:, 0, :, :])
        label_mag, label_phase = torch.norm(label, dim=1), torch.atan2(label[:, -1, :, :],
                                                                            label[:, 0, :, :])
        if feat_type == 'normal':
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'sqrt':
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** 2
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)

        clean_utts, esti_utts = [], []
        for i in range(utt_num):
            tf_esti = esti_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).contiguous().cpu()
            tf_esti = torch.view_as_complex(tf_esti)
            t_esti = torch.istft(tf_esti, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                                 window=torch.hann_window(win_size)).transpose(1, 0).squeeze(dim=-1).numpy()
            tf_label = label_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).contiguous().cpu()
            tf_label = torch.view_as_complex(tf_label)
            t_label = torch.istft(tf_label, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                                  window=torch.hann_window(win_size)).transpose(1, 0).squeeze(dim=-1).numpy()
            t_len = (frame_list[i] - 1) * win_shift
            t_esti, t_label = t_esti[:t_len], t_label[:t_len]
            esti_utts.append(t_esti)
            clean_utts.append(t_label)

        mask_for_loss = []
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()

    loss3 = []
    for id in range(utt_num):
      loss3.append(time_loss(id, esti_utts, clean_utts))
    loss3 = np.mean(loss3)

    return 0.5*loss1 + 0.6*loss2 + 0.2*loss3


def pesq_loss(esti_list, label_list, frame_list):
    with torch.no_grad():
        esti_mag, esti_phase = torch.norm(esti_list, dim=1), torch.atan2(esti_list[:, -1, :, :], esti_list[:, 0, :, :])
        label_mag, label_phase = torch.norm(label_list, dim=1), torch.atan2(label_list[:, -1, :, :],
                                                                            label_list[:, 0, :, :])
        if feat_type == 'normal':
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'sqrt':
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** 2
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'cubic':
            esti_mag = esti_mag ** (10 / 3)
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** (10 / 3)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'log_1x':
            esti_mag = torch.exp(esti_mag) - 1
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = torch.exp(label_mag) - 1
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        clean_utts, esti_utts = [], []
        utt_num = label_list.size()[0]
        for i in range(utt_num):
            tf_esti = esti_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).contiguous().cpu()
            tf_esti = torch.view_as_complex(tf_esti)
            t_esti = torch.istft(tf_esti, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                                 window=torch.hann_window(win_size)).transpose(1, 0).squeeze(dim=-1).numpy()
            tf_label = label_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).contiguous().cpu()
            tf_label = torch.view_as_complex(tf_label)
            t_label = torch.istft(tf_label, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                                  window=torch.hann_window(win_size)).transpose(1, 0).squeeze(dim=-1).numpy()
            t_len = (frame_list[i] - 1) * win_shift
            t_esti, t_label = t_esti[:t_len], t_label[:t_len]
            esti_utts.append(t_esti)
            clean_utts.append(t_label)

        #cv_pesq_score = np.float32([])
        #for id in range(utt_num):
        #  cv_pesq_score = np.append(cv_pesq_score, eval_pesq(id, esti_utts, clean_utts))
        cv_pesq_score = Parallel(n_jobs=1)(delayed(eval_pesq)(id, esti_utts, clean_utts) for id in range(utt_num))
        cv_pesq_score = np.mean(cv_pesq_score)
    return 4.5 - cv_pesq_score


def eval_pesq(id, esti_utts, clean_utts):
    clean_utt = clean_utts[id]
    esti_utt = esti_utts[id]
    pesq_score = pesq(16000, clean_utt, esti_utt, 'wb')
    # pesq_score = pesq(clean_utt, esti_utt, fs=16000)  # pesq_score = pesq(16000, clean_utt, esti_utt, 'wb') #snr(clean_utt, esti_utt, 16000)

    return pesq_score

def time_loss(id, esti_utts, clean_utts):
    clean_utt = clean_utts[id]
    esti_utt = esti_utts[id]
    mse_time = ( (np.array(clean_utt) - np.array(esti_utt)) ** 2 ).mean()

    return mse_time



def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(
            signal_energy / (noise_energy + EPS) + EPS
        )
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return np.mean(segmental_snr)
