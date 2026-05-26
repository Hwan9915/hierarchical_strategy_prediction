"""Strategy (predictor) training. GPU 0. comforting_predictor uses turn=1; all others use turn=5."""
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import strategy_prediction.module as pipeline_module
import strategy_prediction.dataloader_prediction as dataloader
from strategy_prediction.config import MODULE_DICT
from strategy_prediction.model_settings import (
    DEFAULT_MODEL_DIR,
    encoder_slug,
    get_ckpt_dir,
    resolve_encoder,
    turn_for_module,
)
from strategy_prediction.utils import setup_log_file, suppress_training_warnings

STRATEGY_MODULES = [
    "exploration_predictor",
    "comforting_predictor",
    "action_predictor",
]


def train_single(args: argparse.Namespace, module_name: str) -> None:
    turn = turn_for_module(module_name)
    num_labels, _, label_mapping_dict, sampling = MODULE_DICT[module_name]
    train_loader, valid_loader, test_loader = dataloader.create_dataloader(
        sampling=sampling,
        num_labels=num_labels,
        label_mapping_dict=label_mapping_dict,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        test_batch_size=args.test_batch_size,
        turns=turn,
        is_train=1,
    )

    model = pipeline_module.TransformerClassifier(
        encoder_name=args.encoder_name,
        num_labels=num_labels,
        lr=args.lr,
    )
    save_dir = get_ckpt_dir(args.model_dir, module_name, args.encoder_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lightning auto-versions to '{module_name}-v1.ckpt' when the base file exists,
    # but get_ckpt_path() always reads '{module_name}.ckpt' → stale weights at infer time.
    for stale in save_dir.glob(f"{module_name}*.ckpt"):
        stale.unlink()

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename=module_name,
        save_weights_only=True,
    )
    tb_logger = None
    if not args.disable_tb:
        tb_logger = TensorBoardLogger(
            save_dir=args.tb_log_dir,
            name=f"strategy_{module_name}_{encoder_slug(args.encoder_name)}",
        )

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[ckpt_cb],
        enable_progress_bar=bool(args.verbose),
        enable_checkpointing=True,
        enable_model_summary=False,
    )
    if args.demo:
        trainer_kwargs.update(
            max_epochs=1,
            limit_train_batches=0.1,
            limit_val_batches=0.1,
            limit_test_batches=0.1,
        )

    trainer = pl.Trainer(**trainer_kwargs)

    if args.verbose:
        print(f"[Strategy][Train] turn={turn} {module_name} start")
        if tb_logger is not None:
            print(f"[Strategy][TB] log_dir: {tb_logger.log_dir}")
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader, verbose=bool(args.verbose))

    if args.verbose:
        print(f"[Strategy][Train] turn={turn} {module_name} done")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--valid_batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--encoder_key", type=str, default="distilbert")
    p.add_argument("--encoder_name", type=str, default="")
    p.add_argument("--tb_log_dir", type=str, default="strategy_prediction/tb_logs")
    p.add_argument("--disable_tb", type=int, default=1)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--log_file", type=str, default="")
    p.add_argument("--module", type=str, default="", help="Run only this module")
    p.add_argument("--demo", action="store_true", help="Short demo run")
    return p


def main() -> None:
    args = get_parser().parse_args()
    setup_log_file(args.log_file)
    suppress_training_warnings()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    pl.seed_everything(args.seed, workers=True)
    _, resolved_encoder = resolve_encoder(args.encoder_key, args.encoder_name)
    args.encoder_name = resolved_encoder

    if args.verbose:
        print(f"[Strategy][Train] encoder: {args.encoder_name} (demo={args.demo})")

    modules_to_run = [args.module] if args.module else STRATEGY_MODULES
    if args.module and args.module not in STRATEGY_MODULES:
        raise ValueError(f"--module must be one of {STRATEGY_MODULES}, got {args.module}")

    for module_name in modules_to_run:
        train_single(args, module_name)


if __name__ == "__main__":
    main()
