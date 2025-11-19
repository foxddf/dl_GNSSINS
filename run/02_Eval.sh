python ../src/02_TrainINSDrift.py eval \
    --csv ../outputs/synthetic/imu_and_ins.csv \
    --checkpoint ../outputs/artifacts/drift_gru.pt \
    --output-csv ../outputs/artifacts/drift_corrections.csv