from pytorch_lightning.cli import LightningCLI

from classifier import FELDClassifier
from datamodule import FeldDatamodule


if __name__ == '__main__':
    LightningCLI(
        FELDClassifier,
        FeldDatamodule,
        save_config_kwargs={"overwrite": True},
    )