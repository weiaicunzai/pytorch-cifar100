""" dynamically load settings

author baiyu
"""
import conf.global_settings as settings
from conf.argparse import get_args
from conf.experement_setup import get_checkpoint_path
from conf.experement_setup import get_experiment_name


class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(settings)
