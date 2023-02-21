import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple

STAT_ROOT = '../stat_exp_checkpoint'


def stability_score(diag_probs):
    """ Assessment of the stability of the prediction of the model at different image shifts 
    diag_probs:    model prediction confidence at different image shifts 
    """
    return np.var(diag_probs[..., 1:] - diag_probs[..., :-1], axis=2)
    

def stat_probs_from_folder(folder_path: str, mean_by_sampels=True, index=None) -> Tuple[np.ndarray]:
    
    folder_path = Path(folder_path)
    
    # do our correct class prob. more confidence (diff seeds)
    confid_diag_probas = np.stack([
        np.load(m_folder / 'diag_probs.npy')
        for m_folder in sorted(folder_path.iterdir())
        if m_folder.is_dir()
    ])
    
    # if index is None:
    #     stability_scores = stability_score(confid_diag_probas)
    #     index = stability_scores.argmin(1)

    
    confid_diag_probas = confid_diag_probas[:, [12]*3]
    confid_diag_probas = confid_diag_probas.mean(axis=1 if mean_by_sampels else 0)

    np.save(str(folder_path / 'confid_diag_probas.npz'), confid_diag_probas)

    # mean, std by sampels
    mean, std = confid_diag_probas.mean(axis=0), confid_diag_probas.std(axis=0)
        
    return mean / 100., std / 100., index


def plot_exp_consist(folder_path: str, ax: plt.Axes, show_std: bool = False, label=None, index=None):
    
    folder_path = Path(folder_path)
    mean, std, index = stat_probs_from_folder(folder_path, index=index)
    x = np.arange(1, len(mean)+1)
    
    if not label:
        label = folder_path.name
    
    ax.plot(x, mean, label=label)

    if show_std:
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17))
    
    return index


if __name__ == '__main__':


    exp_name1 = Path(STAT_ROOT) / 'resnet18'
    #exp_name2 = Path(STAT_ROOT) / 'resnet18_bss2_rc_aug_1.0_aug_mode_pad_log_no_w'
    exp_name3 = Path(STAT_ROOT) / 'resnet18_lpf3_bss1_rc_aug_1.0_aug_mode_pad_log_no_w'
    exp_name4 = Path(STAT_ROOT) / 'resnet18_bss2_rc_aug_0.9_aug_mode_pad_log_cross_loss_3.0w_start80_temp15_log_no_w'

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_ylabel('Prob of correct class')
    ax.set_xlabel('Diagonal Shift')
    
    show_std = True
    
    index = plot_exp_consist(exp_name1, ax, show_std=show_std)
    plot_exp_consist(exp_name3, ax, show_std=show_std, label="resner18 blurpool", index=index)
    plot_exp_consist(exp_name4, ax, show_std=show_std, label="resner18 cross loss", index=index)
    
    fig.savefig('./full_figure.png', pad_inches=0.2, bbox_inches='tight')
