from pathlib import Path

from conf import settings
from utils import most_recent_folder


def get_experiment_name(args):
    exp_name = args.net

    if args.bp_filt_size:
        exp_name += f"_lpf{args.bp_filt_size}"

    exp_name += f"_{args.dataset}"

    exp_name += f"_x{args.multiply_data}_data"

    if args.orig_augs:
        exp_name += "_orig_augs"
    else:
        exp_name += f"_rc_aug_{args.prob_aug}_aug_mode_{args.mode_aug}"

    if args.use_cross_loss:
        exp_name += f"_log_cross_loss_{args.cross_loss_weight}w"

        if args.cross_loss_start_epoch > 0:
            exp_name += f"_start{args.cross_loss_start_epoch}"

        if args.only_correct_cross_loss:
            exp_name += f"_only_correct"

        if args.soft_temper > 1:
            exp_name += f"_temp{args.soft_temper}"

    if args.use_avg_cross_loss:
        exp_name += f"_log_avg_cross_loss_{args.avg_cross_loss_weight}w"
        if args.avg_cross_loss_start_epoch > 0:
            exp_name += f"_start{args.avg_cross_loss_start_epoch}"

    if args.x2_epoch:
        exp_name += "_x2_epoch"

    exp_name += "_log_no_w"

    return exp_name


def get_checkpoint_path(args, exp_name):
    checkpoint_path = Path(settings.CHECKPOINT_PATH)

    if args.use_distil_aug:
        checkpoint_path = checkpoint_path / 'distil_aug'
        checkpoint_path = checkpoint_path / f'w_{args.distil_aug_weight}' \
                                            f'_func_{args.distil_function}' \
                                            f'_temp_{args.temperature}_1mlstone'

    checkpoint_path = checkpoint_path / exp_name / f"seed{args.seed}"

    if args.resume:
        recent_folder = most_recent_folder(
            str(checkpoint_path),
            fmt=settings.DATE_FORMAT
        )

        checkpoint_path = checkpoint_path / recent_folder

    else:
        checkpoint_path = checkpoint_path / settings.TIME_NOW

    return checkpoint_path
