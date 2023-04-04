
echo "Running train_LPSL CIFAR10"
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/hier/CIFAR10/train_LPSL.py --gpu 1 --_lambda 0 --eta 0.01 > log.out 2>&1 &

echo "Running train_LPSL_acts CIFAR10"
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/hier/CIFAR10/train_LPSL_acts.py \
    --gpu 1 \
    --_lambda 0 \
    --eta 0 \
    --q_budget 2000 \
    --delta F \
    --T1 0.1 \
    --bs 64 \
    --E1 21 \
    --E2 30 \
    > log.out 2>&1 &
