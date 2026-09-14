# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/segformer_rsp.py',
    # Data Loading
    '../_base_/datasets/vaih_irrg2pots_irrg_new.py',
    # UDA Self-Training
    '../_base_/uda/progressive_cotraining_network.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw_40k.py',
]
expr_name='PCN_vaih_irrg2pots_irrg_segformer_semantic'

# Random Seed
seed = 0

model = dict(
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
)

# Call optimizer within train_step
optimizer_config = None

# Modifications to Basic UDA
uda = dict(
    aux_losses=[
        dict(
            type='PFGSTLoss',
            kernel_size=3,
            dilation=2,
            top_k=3,
            weights={'src_pos': 0.1, 'src_neg': 0.1, 'sim_pos': 0.1,
                     'sim_neg': 0.1, 'src_pos_std': 0.1, 'src_neg_std': 0.1},
            sim_type='cosine',
            feat_level=None,
            detach_unfold=True,
            downscale=0.5
        ),
    ],
    alpha=0.999,
    thre_type='all',
    pseudo_threshold=0.98,
    trg_loss_weight=1.,
    use_decoded_feats=True,
)

init_kwargs = dict(
    project='rsi_dass',
    entity='tum-tanmlh',
    name=expr_name,
    resume='never'
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
