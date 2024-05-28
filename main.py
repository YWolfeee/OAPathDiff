from typing import List, Optional, Tuple
from uuid import uuid4
import os
import argparse
import shutil
import torch

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet



def get_config():
    # Convert all hard coding paramters in main.py into an argparse
    # Add the following code to the end of the main.py file
    import argparse
    parser = argparse.ArgumentParser("Train a DDPM model for OAReactDiff")
    # hyperparameters

    # -- running config -- 
    parser.add_argument("--model_type", type=str, default="leftnet", help="Model type to use")
    parser.add_argument("--version", type=str, default="0", help="Model version")
    parser.add_argument("--project", type=str, default="OAReactDiff", help="Project name")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--datadir", type=str, default="oa_reactdiff/data/transition1x/", help="Data directory")
    parser.add_argument("--save_path", type=str, default="working/debug", help="Save path")
    parser.add_argument("--run_name", type=str, default="initial_exp", help="Run name")
    parser.add_argument("--save_top_k", type=int, default=5, help="Save every n epochs")

    # -- training config --
    parser.add_argument("--node_nfs", type=int, default=10, help="Node feature dimensions")
    parser.add_argument("--edge_nf", type=int, default=0, help="Edge feature dimension")
    parser.add_argument("--condition_nf", type=int, default=1, help="Condition feature dimension")
    parser.add_argument("--update_pocket_coords", type=bool, default=True, help="Update pocket coordinates")
    parser.add_argument("--condition_time", type=bool, default=True, help="Condition on time")
    parser.add_argument("--edge_cutoff", type=float, default=None, help="Edge cutoff")
    parser.add_argument("--loss_type", type=str, default="l2", help="Loss type")
    parser.add_argument("--pos_only", type=bool, default=True, help="Position only")
    parser.add_argument("--process_type", type=str, default="TS1x", help="Process type")
    parser.add_argument("--enforce_same_encoding", type=bool, default=None, help="Enforce same encoding")
    parser.add_argument("--eval_epochs", type=int, default=10, help="Evaluation epochs")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="Noise schedule")
    parser.add_argument("--timesteps", type=int, default=5000, help="Timesteps")
    parser.add_argument("--precision", type=float, default=1e-5, help="Precision")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--clip_grad", type=bool, default=True, help="Clip gradient")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clip value")
    parser.add_argument("--append_frag", type=bool, default=False, help="Append fragment")
    parser.add_argument("--use_by_ind", type=bool, default=True, help="Use by index")
    parser.add_argument("--reflection", type=bool, default=False, help="Reflection")
    parser.add_argument("--single_frag_only", type=bool, default=True, help="Single fragment only")
    parser.add_argument("--only_ts", type=bool, default=False, help="Only TS")

    parser.add_argument("--append_t", type=bool, default=True, help="Append time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Max epochs")
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args


def get_model(args: argparse.Namespace):
    # ---EGNNDynamics---
    egnn_config = dict(
        in_node_nf=8,  # embedded dim before injecting to egnn
        in_edge_nf=0,
        hidden_nf=256,
        edge_hidden_nf=64,
        act_fn="swish",
        n_layers=9,
        attention=True,
        out_node_nf=None,
        tanh=True,
        coords_range=15.0,
        norm_constant=1.0,
        inv_sublayers=1,
        sin_embedding=True,
        normalization_factor=1.0,
        aggregation_method="mean",
    )
    leftnet_config = dict(
        pos_require_grad=False,
        cutoff=10.0,
        num_layers=6,
        hidden_channels=196,
        num_radial=96,
        in_hidden_channels=8,
        reflect_equiv=True,
        legacy=True,
        update=True,
        pos_grad=False,
        single_layer_output=True,
        object_aware=True,
    )

    if args.model_type == "leftnet":
        model_config = leftnet_config
        model = LEFTNet
    elif args.model_type == "egnn":
        model_config = egnn_config
        model = EGNN
    else:
        raise KeyError("model type not implemented.")
    
    return model, model_config

def get_diffusion(args: argparse.Namespace, 
                  model: torch.nn.Module,
                  model_config: dict,):
    optimizer_config = dict(
        lr=2.5e-4,
        betas=[0.9, 0.999],
        weight_decay=0,
        amsgrad=True,
    )

    training_config = dict(
        datadir=args.datadir,
        remove_h=False,
        bz=args.batch_size,
        num_workers=0,
        clip_grad=args.clip_grad,
        gradient_clip_val=args.gradient_clip_val,
        ema=False,
        ema_decay=0.999,
        swapping_react_prod=True,
        append_frag=args.append_frag,
        use_by_ind=args.use_by_ind,
        reflection=args.reflection,
        single_frag_only=args.single_frag_only,
        only_ts=args.only_ts,
        lr_schedule_type=None,
        lr_schedule_config=dict(
            gamma=0.8,
            step_size=100,
        ),  # step
        append_t=args.append_t,
    )

    scales = [1.0, 2.0, 1.0]
    fixed_idx: Optional[List] = None

    # ----Normalizer---
    norm_values: Tuple = (1.0, 1.0, 1.0, 1.0)
    norm_biases: Tuple = (0.0, 0.0, 0.0, 0.0)


    earlystopping = EarlyStopping(
        monitor="val-totloss",
        patience=2000,
        verbose=True,
        log_rank_zero_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [earlystopping, TQDMProgressBar(), lr_monitor]
    if training_config["ema"]:
        callbacks.append(EMACallback(decay=training_config["ema_decay"]))


    ddpm = DDPMModule(
        model_config,
        optimizer_config,
        training_config,
        model=model,
        node_nfs=[args.node_nfs] * 3,
        edge_nf=args.edge_nf,
        condition_nf=args.condition_nf,
        fragment_names=["R", "TS", "P"],
        pos_dim=3,
        update_pocket_coords=args.update_pocket_coords, 
        condition_time=args.condition_time, 
        edge_cutoff=args.edge_cutoff, 
        norm_values=norm_values,
        norm_biases=norm_biases,
        noise_schedule=args.noise_schedule, 
        timesteps=args.timesteps, 
        precision=args.precision, 
        loss_type=args.loss_type,
        pos_only=args.pos_only,
        process_type=args.process_type,
        enforce_same_encoding=args.enforce_same_encoding,
        eval_epochs=args.eval_epochs,
        scales=scales,
        source=None,
        fixed_idx=fixed_idx,
    )

    training_config.update(optimizer_config)

    return ddpm, training_config, callbacks

if __name__ == "__main__":

    args = get_config()
    seed_everything(args.seed, workers=True)

    model, model_config = get_model(args)
    ddpm, training_config, callbacks = get_diffusion(args, model, model_config)


    config = model_config.copy()
    config.update(training_config)

    print("config: ", config)


    # run_name = f"{args.model_type}-{args.version}-" + str(uuid4()).split("-")[-1]
    run_name = args.run_name + f"+max_epochs={args.max_epochs}+append_t={args.append_t}"
    if args.use_wandb:
        logger = WandbLogger(
            project=args.project,
            log_model=False,
            name=run_name,
        )
    else:
        logger = CSVLogger(args.save_path, name=run_name)
    try:  # Avoid errors for creating wandb instances multiple times
        logger.experiment.config.update(config)
        logger.watch(ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False)
    except:
        pass


    checkpoint_callback = ModelCheckpoint(
        monitor="val-totloss",
        dirpath=logger.log_dir,
        filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
        every_n_epochs=1,
        save_top_k=args.save_top_k,
    )
    callbacks = [checkpoint_callback] + callbacks

    print("logger.log_dir: ", logger.log_dir)

    # if not os.path.isdir(logger.log_dir):
    #     os.makedirs(logger.log_dir)
    # shutil.copy(f"oa_reactdiff/model/{args.model_type}.py", f"{logger.log_dir}/{args.model_type}.py")


    devices = list(range(torch.cuda.device_count()))
    strategy = DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else None

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        deterministic=False,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=1,
        callbacks=callbacks,
        profiler=None,
        logger=logger,
        accumulate_grad_batches=1,
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_train_batches=200,
        limit_val_batches=20,
        # max_time="00:10:00:00",
    )

    trainer.fit(ddpm)

    trainer.save_checkpoint(os.path.join(logger.log_dir, "pretrained-ts1x-diff.ckpt"))