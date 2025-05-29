#!/usr/bin/env bash
set -euo pipefail

# 2. 定位 toml
DATASET=${1:-"cifar10"}
MODEL=${2:-"resnet18"}
ALPHA=${3:-"1.0"}
PARAM_FILE="dataset/configurations/${DATASET}_${MODEL}.toml"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "✖ 找不到参数文件：$PARAM_FILE"
  exit 1
fi

echo "▶ 读取参数：${PARAM_FILE}"

# 3. 用 Python 加载 TOML，并输出 key=value 列表
mapfile -t KV_LINES < <(python3 - <<PYCODE
import toml, sys
cfg = toml.load("$PARAM_FILE")
cfg["alpha"] = $ALPHA
for key, val in cfg.items():
    if isinstance(val, str):
        print(f'{key}="{val}"')
    # 强制格式化浮点、整数为字符串
    else:
        print(f"{key}={val}")
PYCODE
)

# 如果没读到任何行，就报错
if (( ${#KV_LINES[@]} == 0 )); then
    echo "✖ 参数文件里没有任何键值对！"
    exit 1
fi

# 4. 构造 CLI 覆盖参数数组
RUN_CONFIG_ARGS=()
for kv in "${KV_LINES[@]}"; do
    echo "KV: $kv"
    RUN_CONFIG_ARGS+=( "$kv" )
done

# 5. 把数组变成单条字符串
OVERRIDE_STR="${RUN_CONFIG_ARGS[*]}"

# 6. 直接传给 Flower——注意这里只有双引号，没有单引号
echo "▶ 启动 Flower，使用 GPU 模拟"
echo "  flwr run . --run-config \"$OVERRIDE_STR\""
flwr run . \
    --run-config "$OVERRIDE_STR"
