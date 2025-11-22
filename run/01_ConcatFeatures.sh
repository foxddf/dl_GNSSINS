# 헤더는 첫 파일만 쓰고 나머지는 헤더 제거 후 이어붙이는 방식
cd ../outputs/synthetic_multi

# 첫 파일의 헤더 + 내용
head -n 1 scenario_000/bias_training.csv > all_bias_training.csv
# 나머지 파일들은 헤더 제외하고 붙이기
for d in scenario_*; do
    tail -n +2 "$d/bias_training.csv" >> all_bias_training.csv
done
