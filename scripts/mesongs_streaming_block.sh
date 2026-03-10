DATASET=db
SCENE=drjohnson
ITERS=0 # 0 for compression without finetuning.
DATAPATH=/synology/rajrup/MesonGS/data/$DATASET/$SCENE
INITIALPATH=/synology/rajrup/MesonGS/train_output/$DATASET/$SCENE/point_cloud/iteration_30000/point_cloud.ply
CONFIG=config3
CSVPATH=/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression/mesongs/streaming\_$CONFIG.csv
SAVEPATH=/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression/mesongs/streaming\_$CONFIG

LSEG=0 # using the pre-written config, so do not use the LSED config.
CB=0 # same as LSEG
DEPTH=0 # same as LSEG

mkdir -p /synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression
CUDA_VISIBLE_DEVICES=1 python mesongs_streaming.py -s $DATAPATH \
    --given_ply_path $INITIALPATH \
    --num_bits 8 \
    --convert_SHs_python \
    --percent 0 \
    --prune \
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
    --eval \
    --output_path $SAVEPATH \
    --save_renders
