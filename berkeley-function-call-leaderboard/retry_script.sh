#!/bin/bash

# 重试次数计数
attempt=0

# 最大重试次数
max_attempts=10000

# 重试间隔时间（秒）
retry_interval=1

# 命令
command="python openfunctions_evaluation.py --model ernie-3.5-8k-0205 --test-category all"

until [ $attempt -ge $max_attempts ]
do
  $command && break  # 如果命令成功，则退出循环
  attempt=$((attempt+1))
  echo "Attempt $attempt failed! Retrying in $retry_interval seconds..."
  sleep $retry_interval
done

if [ $attempt -ge $max_attempts ]; then
  echo "Reached maximum attempts. Exiting..."
fi
