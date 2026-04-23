#!/bin/bash
# 打包脚本 - 生成遵循 .gitignore 的部署包

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="wechatdoc_finetune_${TIMESTAMP}.tar.gz"

echo "============================================================"
echo "  项目打包脚本"
echo "============================================================"
echo "工作目录: $SCRIPT_DIR"
echo "输出文件: $PACKAGE_NAME"
echo ""

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "错误: 当前目录不是 git 仓库"
    exit 1
fi

# 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo "警告: 存在未提交的更改"
    echo "建议先提交更改，或按 Ctrl+C 取消"
    echo ""
    read -p "是否继续打包? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 打包
echo "正在打包..."
git archive --format=tar.gz --output="$PACKAGE_NAME" HEAD

if [ $? -eq 0 ]; then
    SIZE=$(ls -lh "$PACKAGE_NAME" | awk '{print $5}')
    echo ""
    echo "✓ 打包成功: $PACKAGE_NAME ($SIZE)"
    echo ""
    echo "上传到服务器示例:"
    echo "  scp $PACKAGE_NAME user@server:/path/to/destination/"
    echo ""
    echo "服务器端解压:"
    echo "  tar -xzf $PACKAGE_NAME"
    echo ""
    echo "注意: 需要单独上传 Excel 数据文件"
    echo "  scp *.xlsx user@server:/path/to/project/"
else
    echo "✗ 打包失败"
    exit 1
fi
