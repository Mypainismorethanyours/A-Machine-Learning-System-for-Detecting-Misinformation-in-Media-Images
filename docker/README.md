```
git clone --recurse-submodules https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images"

docker build -t jupyter-mlflow -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/Dockerfile.jupyter-torch-mlflow-cuda .

docker compose -f A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/docker-compose-mlflow.yaml up -d

HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)

docker run -it --rm --gpus all -p 8888:8888 --shm-size=8g -v $(pwd):/workspace -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000 jupyter-mlflow
```
