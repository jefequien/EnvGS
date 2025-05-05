#!/bin/bash 
set -xe

#OAR -q production 
#OAR -l host=1/gpu=2,walltime=24:00:00
#OAR -p gpu_model='H100 NVL'
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

module load cuda/11.8.0_gcc-10.4.0

# make use of a python torch environment
source ~/.bashrc
conda activate envgs
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";

# DATASET_NAME="ref_real"
# SCENE_LIST="sedan toycar spheres"
# TRAJ_NAME="spiral"

# DATASET_NAME="360_v2"
# # SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# # SCENE_LIST="bicycle stump treehill flowers"
# SCENE_LIST="bonsai counter kitchen room"
# TRAJ_NAME="spiral"

DATASET_NAME="neural_catacaustics"
# SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
SCENE_LIST="wateringcan2"
TRAJ_NAME="spiral"

# DATASET_NAME="renders"
# # SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
# SCENE_LIST="shiny_office shiny_bedroom"
# TRAJ_NAME="spiral"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # Train
    evc-train -c configs/exps/envgs/$DATASET_NAME/envgs_$SCENE.yaml exp_name=envgs/$DATASET_NAME/envgs_$SCENE

    # Move checkpoint and render novel views
    mkdir -p data/trained_model/envgs_$SCENE
    cp data/trained_model/envgs/$DATASET_NAME/envgs_$SCENE/latest.npz  data/trained_model/envgs_$SCENE/latest.npz 
    evc-test -c configs/exps/envgs/$DATASET_NAME/envgs_$SCENE.yaml,configs/specs/$TRAJ_NAME.yaml

    # Saving videos
    IMAGES_DIR=data/novel_view/envgs_$SCENE
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/RENDER/*.png"   -c:v libx264 -pix_fmt yuv420p "$IMAGES_DIR/RENDER.mp4"
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/DIFFUSE/*.png"   -c:v libx264 -pix_fmt yuv420p "$IMAGES_DIR/DIFFUSE.mp4"
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/REFLECTION/*.png"   -c:v libx264 -pix_fmt yuv420p "$IMAGES_DIR/REFLECTION.mp4"
    ffmpeg -i "$IMAGES_DIR/RENDER.mp4" -i "$IMAGES_DIR/DIFFUSE.mp4" -i "$IMAGES_DIR/REFLECTION.mp4" -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" "$IMAGES_DIR/RENDER,DIFFUSE,REFLECTION.mp4"

done
