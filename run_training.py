from pytorch_lightning.cli import LightningCLI

from classifier import FELDClassifier
from peld_datamodule import PeldDatamodule


if __name__ == '__main__':
    cli = LightningCLI(
        FELDClassifier,
        save_config_kwargs={"overwrite": True},
        run=False,
    )

    cli.trainer.fit(
        model=cli.model, 
        datamodule=cli.datamodule,
    )

    cli.trainer.test(
        model=cli.model, 
        datamodule=cli.datamodule,
        ckpt_path='best',
    )