echo "Running train_LPSL KFPI"
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/hier/KFPI/train_LPSL.py --gpu 0 --_lambda 10 --eta 10 > log.out 2>&1 &

echo "Running train_LPSL_acts KFPI"
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/hier/KFPI/train_LPSL_acts.py \
    --gpu 0 \
    --_lambda 0 \
    --eta 0 \
    --q_budget 2000 \
    --delta F \
    --T1 0.1 \
    --bs 64 \
    --E1 16 \
    --E2 20 \
    > log.out 2>&1 &