import os
import torch
import argparse
import numpy as np
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Models.diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=2024,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--sample_num', type=int, default=None,
                        help='Number of windows to generate. None = full non-overlap dataset.')
    parser.add_argument('--sample_batch_size', type=int, default=400,
                        help='GPU batch size per diffusion reverse pass')
    parser.add_argument('--milestone', type=int, default=1000)
    parser.add_argument(
        '--sampling_mode',
        type=str,
        default='ordered_non_overlapping',
        choices=['random', 'ordered', 'ordered_non_overlapping'],
    )
    parser.add_argument('--opts', nargs='+', default=None,
                        help='Optional config overrides, e.g. dataloader.train_dataset.params.proportion 1.0')

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()

    if args.train:
        dataloader_info = build_dataloader(config, args)
    else:
        dataloader_info = {'dataloader': [], 'dataset': None}

    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    else:
        trainer.load(args.milestone)

        sampling_dataset_config = config['dataloader']['train_dataset'].copy()
        sampling_dataset_config['params']['proportion'] = 0.0
        sampling_dataset_config['params']['style'] = 'non_overlapping'
        sampling_dataset_config['params']['save2npy'] = False
        sampling_dataset_config['params']['period'] = 'test'
        sampling_dataset = instantiate_from_config(sampling_dataset_config)

        dataset = sampling_dataset
        max_windows = len(dataset)
        print(f"Non-overlapping sampling dataset: {max_windows} windows")

        if args.sampling_mode == 'ordered_non_overlapping':
            stride = 1
            ordered = True
            num_samples = args.sample_num if args.sample_num is not None else max_windows
            if num_samples > max_windows:
                print(f"Warning: requested {num_samples} windows, dataset fits {max_windows} non-overlapping.")
        elif args.sampling_mode == 'ordered':
            stride = 1
            ordered = True
            num_samples = args.sample_num if args.sample_num is not None else max_windows
        else:
            stride = 1
            ordered = False
            num_samples = args.sample_num if args.sample_num is not None else min(2500, max_windows)

        print(f"Sampling mode: {args.sampling_mode}")
        print(f"Generating {num_samples} windows (batch_size={args.sample_batch_size})")

        samples = trainer.sample(
            num=num_samples,
            size_every=args.sample_batch_size,
            shape=[dataset.window, dataset.var_num],
            dataset=dataset,
            ordered=ordered,
            stride=stride,
        )

        if dataset.auto_norm:
            n, length, channels = samples.shape
            if channels == 9:
                power = unnormalize_to_zero_to_one(samples[:, :, 0:1])
                time_feats = samples[:, :, 1:9]
                power_flat = power.reshape(-1, 1)
                power_watts = dataset.scaler.inverse_transform(power_flat).reshape(n, length, 1)
                samples = np.concatenate([power_watts, time_feats], axis=2)
            else:
                samples = unnormalize_to_zero_to_one(samples)
                samples_flat = samples.reshape(-1, channels)
                samples = dataset.scaler.inverse_transform(samples_flat).reshape(n, length, channels)

            print(f"Generated shape: {samples.shape}")
            print(f"Power range (W): {samples[:, :, 0].min():.2f} to {samples[:, :, 0].max():.2f}")
            np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)


if __name__ == '__main__':
    main()
