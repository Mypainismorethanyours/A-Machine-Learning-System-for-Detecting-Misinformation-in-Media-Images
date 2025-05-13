sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%

sudo mkfs.ext4 /dev/vdb1

sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block

sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block
df -h
