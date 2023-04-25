#!/usr/bin/bash -l
#SBATCH --output=test-model-real-data.out
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH -N 2
#SBATCH -J CVIA-Test-REAL-Stream-2-Elliott
#SBATCH --mail-type=ALL
#SBATCH --mail-user=001@student.uni.lu

cd /home/users/ewobler/satellite-pose-estimation-main/
conda activate torchit

export MODULEPATH=/opt/apps/resif/iris/2019b/default/modules/all/
module load lang/Anaconda3/2020.02
export ANACONDA=/opt/apps/resif/iris/2019b/default/modules/all/lang/Anaconda3/
export MODULEPATH=/opt/apps/resif/iris/2019b/gpu/modules/all/
module load system/CUDA/

# load the pre-trained model and run against "real" test set
python run_train_model.py -l -tr
