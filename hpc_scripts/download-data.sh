#!/usr/bin/bash -l
#SBATCH --output=download-data.out
#SBATCH --time=06:00:00
#SBATCH -J download-spark-data

cd /home/users/ewobler/streamdata

srun wget -O stream2.zip 'https://uniluxembourg-my.sharepoint.com/:u:/g/personal/arunkumar_rathinam_uni_lu/EWbZeKrI3u1ErkTP01Ws0L8Bd6jfCTkApDVJlZDW_ITrig?download=1'