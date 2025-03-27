# excecuting sequential finetuning on sandbox platform

declare -a datasets=(
    "google_robot_close_bottom_drawer"
    "google_robot_close_middle_drawer"
    "google_robot_close_top_drawer"
    "google_robot_open_bottom_drawer"
    "google_robot_open_middle_drawer"
    "google_robot_open_top_drawer"
)

SperateTrain(){
    echo "SperateTrain"
    python CL-exp/run_code_sandbox.py --config_path=${config_path} --dataset_name=${dataset}
    echo "SperateTrain Done"
}

config_path="/home/v-qilinzhang/CL-exp/configs/sandbox/dora_separate_baseline.yaml"

# draccus allows overwriting parameters in the config file with command line arguments
for dataset in "${datasets[@]}"; do
    echo "dataset: ${dataset}"
    SperateTrain
done

echo "All Done"