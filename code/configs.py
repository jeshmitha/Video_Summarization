import argparse
from pathlib import Path
import pprint

save_dir = Path('./saved_files')

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.setting)

    def set_dataset_dir(self, setting):
        self.train_score_dir = save_dir.joinpath(setting, 'results/train/split'+str(self.split_index))
        self.test_score_dir = save_dir.joinpath(setting, 'results/test/split'+str(self.split_index))
        self.model_dir = save_dir.joinpath(setting, 'models/split'+str(self.split_index))

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--setting', type=str, default='tvsum_canon')

    # Model
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--leaky_relu_negative_slope', type=float, default=0.2)
    parser.add_argument('--regularization_factor', type=float, default=0.15)

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--frame_scorer_lr', type=float, default=1e-5)
    parser.add_argument('--generator_lr', type=float, default=1e-7)
    parser.add_argument('--discriminator_lr', type=float, default=1e-7)
    parser.add_argument('--split_index', type=int, default=0)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)