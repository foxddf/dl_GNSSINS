python ../src/02_TrainINSDrift.py train \
    --csv ../outputs/synthetic/imu_and_ins.csv \
    --output ../outputs/artifacts/drift_gru.pt \
    --window-size 200 \
    --stride 5 \
    --batch-size 64 \
    --val-ratio 0.2 \
    --epochs 100 \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.1 \
    --lr 1e-4 \
    --grad-clip 1.0 \
    --seed 42

python ../src/02_TrainINSDrift.py eval \
    --csv ../outputs/synthetic/imu_and_ins.csv \
    --checkpoint ../outputs/artifacts/drift_gru.pt \
    --output-csv ../outputs/artifacts/drift_corrections.csv