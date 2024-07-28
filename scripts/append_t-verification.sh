#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --save_path "working/newlog-t-verification" --append_t "1" --max_epochs 500 --single_frag_only "" --datadir "oa_reactdiff/data/transition1x/train_addprop.pkl" --process_type "TS1x" --run_name "check_switching_set" --use_by_ind "" > logs/newlog-t-verification+notsingle+process_type=TS1x+epochs=500+append_t=True.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --save_path "working/newlog-t-verification" --append_t "1" --max_epochs 500 --single_frag_only "" --datadir "oa_reactdiff/data/path/path_structures.npz" --process_type "Path" --run_name "check_switching_set" --use_by_ind "" --only_ts "" > logs/newlog-t-verification+notsingle+process_type=Path+epochs=500+append_t=True.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python -u main.py --save_path "working/newlog-t-verification" --append_t "1" --max_epochs 500 --single_frag_only "" --datadir "oa_reactdiff/data/path/path_structures.npz" --process_type "Path" --run_name "check_switching_set" --use_by_ind "" --only_ts "1" > logs/newlog-t-verification+notsingle+process_type=Path+epochs=500+append_t=True+only_ts=True.log 2>&1 &

# Re-run varying t for the first time
CUDA_VISIBLE_DEVICES=2,3 nohup python -u main.py --save_path "working/newlog-t-verification" --append_t "1" --max_epochs 500 --single_frag_only "" --datadir "oa_reactdiff/data/path/path_structures.npz" --process_type "Path" --run_name "varying_t" --use_by_ind "" --only_ts "" > logs/varying_t+process_type=Path+epochs=500+append_t=True.log 2>&1 &

wait
echo "All done"