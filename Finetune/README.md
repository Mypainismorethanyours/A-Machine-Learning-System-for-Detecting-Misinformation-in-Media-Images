Provision one bare-metal node with 2×Nvidia GPUs under CHI@TACC using 1_create_server.ipynb.

Data preparation
```
# run on node
curl https://rclone.org/install.sh | sudo bash
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
mkdir -p ~/.config/rclone
nano  ~/.config/rclone/rclone.conf
```
```
# Paste the following into the config file.
[chi_tacc]
type = swift
user_id = 791358081c320e1a938257e18fd3279d015b04951b0db939c7cd7c241311e4a3
application_credential_id = 549bb7f422df4748a9a10d380a12a3da
application_credential_secret = 4WfcSzfWLz6IZnYtawYex14j2daKMYcJvQ7XR5h-dTmQ230SbJFK6NIs7TbN-qTwdvgvw0SbtTfmj2oZ3Gwtfw
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
```
```
# run on node
mkdir -p ~/data-persist-chi/ammeba
cd ~/data-persist-chi/ammeba
curl -L https://raw.githubusercontent.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/main/Data-Pipeline/Object-CHI@TACC/Retrieve-Dataset-old/ammeba-etl.yaml -o ammeba-etl.yaml
docker compose -f ammeba-etl.yaml run extract-data
docker compose -f ammeba-etl.yaml run load-data
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object
rclone mount chi_tacc:object-persist-project-5 /mnt/object --read-only --allow-other --daemon
cd
```
MLflow Only
```
git clone --recurse-submodules https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images"
docker build -t jupyter-mlflow -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/Finetune/docker/Dockerfile.jupyter-torch-mlflow-cuda .
docker compose -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/Finetune/docker/docker-compose-mlflow-cuda.yaml up -d
# HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
# In place of A.B.C.D, substitute the floating IP address associated with your Kubernetes deployment.
docker run -it --rm --gpus all -p 8888:8888 --shm-size=16g -v $(pwd):/workspace -v /mnt/object:/mnt/object -e DATA_DIR=/mnt/object -e MLFLOW_TRACKING_URI=http://A.B.C.D:8000 --name jupyter jupyter-mlflow
```
```
# run in a terminal inside jupyter container
cd A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/Finetune
python to_Json.py
pip install -r requirements.txt
deepspeed --num_gpus=2 train_multi_GPUs_LoRA_Full.py
deepspeed --num_gpus=2 train_multi_GPUs_LoRA_Sample.py
deepspeed --num_gpus=1 train_single_GPU_LoRA_Full.py
deepspeed --num_gpus=1 train_single_GPU_LoRA_Sample.py
```
MLflow + Ray
```
# run on node
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/Finetune/docker/docker-compose-mlflow-ray-cuda.yaml up -d
docker build -t jupyter-mlflow-ray -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/Finetune/docker/Dockerfile.jupyter-torch-mlflow-ray-cuda .
# HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
# In place of A.B.C.D, substitute the floating IP address associated with your Kubernetes deployment.
docker run -it --rm --gpus all \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    -v /mnt/object:/mnt/object \
    -e DATA_DIR=/mnt/object \
    -e RAY_ADDRESS=http://A.B.C.D:8265/ \
    -e MLFLOW_TRACKING_URI=http://A.B.C.D:8000/ \
    --name jupyter \
    jupyter-mlflow-ray
```
```
# run in a terminal inside jupyter container
cd A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/workspace_ray
python to_Json.py
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- deepspeed --num_gpus=1 train_single_GPU_LoRA_Sample_Ray.py
```
