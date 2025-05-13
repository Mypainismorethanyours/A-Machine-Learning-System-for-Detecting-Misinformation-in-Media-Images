MLflow Only
```
git clone --recurse-submodules https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images"
docker build -t jupyter-mlflow -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/Dockerfile.jupyter-torch-mlflow-cuda .
docker compose -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/docker-compose-mlflow.yaml up -d
# HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
# In place of A.B.C.D, substitute the floating IP address associated with your Kubernetes deployment.
docker run -it --rm --gpus all -p 8888:8888 --shm-size=16g -v $(pwd):/workspace -e MLFLOW_TRACKING_URI=http://A.B.C.D:8000 --name jupyter jupyter-mlflow
```
MLflow + Ray
```
# run on node
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/docker-compose-ray-cuda.yaml up -d
docker build -t jupyter-ray -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/Dockerfile.jupyter-ray .
# HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
# In place of A.B.C.D, substitute the floating IP address associated with your Kubernetes deployment.
docker run -it --rm --gpus all \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    -v /mnt/object:/mnt/object \
    -e DATA_DIR=/mnt/object \
    -e RAY_ADDRESS=http://A.B.C.D:8082/ \
    -e MLFLOW_TRACKING_URI=http://A.B.C.D:8000/ \
    --name jupyter \
    jupyter-ray
```
```
# run in a terminal inside jupyter container
cd A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/workspace_ray
python to_Json.py
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- deepspeed --num_gpus=1 train_single_GPU_LoRA_Sample.py
```
