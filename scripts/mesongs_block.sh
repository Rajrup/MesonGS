# SCENE=mic
DATASET=db
SCENE=drjohnson
ITERS=0 # 0 for compression without finetuning.
DATAPATH=/home/rajrup/Project/MesonGS/data/$DATASET/$SCENE
INITIALPATH=/home/rajrup/Project/MesonGS/output/$DATASET/$SCENE/point_cloud/iteration_30000/point_cloud.ply
CONFIG=config3
CSVPATH=/home/rajrup/Project/MesonGS/exp_data/csv/$DATASET/$SCENE\_$CONFIG.csv
SAVEPATH=/home/rajrup/Project/MesonGS/output/$DATASET/$SCENE\_$CONFIG

LSEG=0 # using the pre-written config, so do not use the LSED config.
CB=0 # same as LSEG
DEPTH=0 # same as LSEG

mkdir -p /home/rajrup/Project/MesonGS/exp_data/csv/$DATASET/
CUDA_VISIBLE_DEVICES=0 python mesongs.py -s $DATAPATH \
    --given_ply_path $INITIALPATH \
    --num_bits 8 \
    --save_imp --eval \
    --iterations $ITERS \
    --finetune_lr_scale 1 \
    --convert_SHs_python \
    --percent 0 \
    --codebook_size $CB \
    --steps 1000 \
    --scene_imp $SCENE \
    --depth $DEPTH \
    --raht \
    --clamp_color \
    --per_block_quant \
    --lseg $LSEG \
    --use_indexed \
    --debug \
    --hyper_config $CONFIG \
    --csv_path $CSVPATH \
    --model_path $SAVEPATH
