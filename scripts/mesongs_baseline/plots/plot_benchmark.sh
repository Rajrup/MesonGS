#!/bin/bash

# Plot benchmark results for MesonGS compression
DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
CONFIG_NAME="params_default"

working_dir="/home/rajrup/Project/VideoGS"
data_path="/synology/rajrup/VideoGS"
input_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/mesongs"
plot_script_folder="${working_dir}/scripts/mesongs_baseline/plots"

# Plot compressed size + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}

# Plot encode/decode time (stacked area)
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}

# Plot quality (PSNR / SSIM)
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}
