It's the old way to retrive data from Internet with ammeba


[chi_tacc]
type = swift
user_id = 791358081c320e1a938257e18fd3279d015b04951b0db939c7cd7c241311e4a3
application_credential_id = 549bb7f422df4748a9a10d380a12a3da
application_credential_secret = 4WfcSzfWLz6IZnYtawYex14j2daKMYcJvQ7XR5h-dTmQ230SbJFK6NIs7TbN-qTwdvgvw0SbtTfmj2oZ3Gwtfw
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC

mkdir -p ~/data-persist-chi/ammeba
cd ~/data-persist-chi/ammeba

curl -L https://raw.githubusercontent.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/main/Data-Pipeline/Object-CHI@TACC/ammeba-etl.yaml -o ammeba-etl.yaml

docker compose -f ammeba-etl.yaml run extract-data
docker compose -f ammeba-etl.yaml run load-data

rclone tree chi_tacc:object-persist-project-5 --max-depth 2

Cleaning up existing contents in object-persist-project-5 ...
Uploading final_dataset/ to object store...
Transferred:        1.517 GiB / 1.517 GiB, 100%, 4.249 MiB/s, ETA 0s
Transferred:        10267 / 10267, 100%
Elapsed time:      5m25.1s
Verifying structure:
/
└── final_dataset
    ├── images
    ├── test.csv
    ├── train.csv
    └── val.csv



rclone mount chi_tacc:object-persist-project-5 /mnt/object --read-only --allow-other --daemon

ls /mnt/object

docker run -d --rm \
  -p 8888:8888 \
  --shm-size 8G \
  -e AMMeBa_DATA_DIR=/mnt/final_dataset \
  -v ~/data-persist-chi/workspace:/home/jovyan/work/ \
  --mount type=bind,source=/mnt/object,target=/mnt/final_dataset,readonly \
  --name jupyter \
  quay.io/jupyter/pytorch-notebook:latest
