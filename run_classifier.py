from tensorboardX import SummaryWriter

import pytorch_lightning as pl
from models import LitClassifier

if __name__ == "__main__":
    model = LitClassifier(args.learning_rate, transform_func=transform_func, transform_val=args.transform_val, noise=args.noise,
                        nurd=nurd,
                        nurd_balance=args.nurd_balance,
                        hosp_predict=args.hosp_predict)

    # ------------
    # data
    # ------------
    # dm = MNISTDataModule.from_argparse_args(args)
    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='pl_logs_pixelcnn/{}'.format(args.prefix),
    )

    # ------------
    # model
    # ------------
    

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, progress_bar_refresh_rate=0,
        # limit_train_batches=0.2, 
        # limit_val_batches=0.2, 
        # precision=16
        )
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)

    # ------------
    # testing
    # ------------
    # result = trainer.test(model, test_dataloaders=test_loader, limit_test)
    # print(result)


if __name__ == '__main__':
    # cli_lightning_logo()
    cli_main()