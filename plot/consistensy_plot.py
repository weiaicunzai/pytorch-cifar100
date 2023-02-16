import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


STAT_ROOT = '../stat_exp_checkpoint'
    

def stat_probs_from_folder(folder_path: str) -> Tuple[np.ndarray]:
    
    means = []
    folder_path = Path(folder_path)
    
    for m_folder in sorted(folder_path.iterdir()):
        if m_folder.is_dir():
            mean = np.load(m_folder / 'diag_probs.npy').mean(axis=0)
            means.append(mean)
    
    means = np.vstack(means)
    mean, std = means.mean(axis=0), means.std(axis=0)
        
    return mean / 100., std / 100.


def plot_exp_consist(folder_path: str, ax):
    
    mean, std = stat_probs_from_folder(folder_path)
    x = np.arange(1, len(mean)+1)
    
    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-exp-name', type=str, required=True)
    
    args = parser.parse_args()
    exp_name1 = Path(STAT_ROOT) / 'resnet18_bss2_rc_aug_0.9_aug_mode_pad_log_cross_loss_3.0w_start80_temp15_log_no_w'
    exp_name2 = Path(STAT_ROOT) / 'resnet18_bss2_rc_aug_0.9_aug_mode_pad_log_cross_loss_0.0w_start20_log_no_w'
    exp_name3 = Path(STAT_ROOT) / 'resnet18_bss1_rc_aug_1.0_aug_mode_pad_log_no_w'
        
    fig, ax = plt.subplots()
    plot_exp_consist(exp_name1, ax)
    plot_exp_consist(exp_name2, ax)
    plot_exp_consist(exp_name3, ax)
    
    fig.savefig('./full_figure.png')
