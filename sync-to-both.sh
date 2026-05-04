#!/bin/bash
# 快捷配置：将项目同步到 Gitee 和 GitHub
# 用法: ./sync-to-both.sh <gitee用户名> <github用户名>

GITEE_USER=${1:-hanslzh}
GITHUB_USER=${2:-HansTechX}

# 获取当前仓库名
REPO_NAME=$(basename -s .git $(git config --get remote.origin.url))
echo "当前项目: $REPO_NAME"

# 添加双推送地址
git remote set-url --add --push origin git@gitee.com:${GITEE_USER}/${REPO_NAME}.git
git remote set-url --add --push origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git

echo "配置完成！"
echo ""
echo "当前远程仓库配置："
git remote -v
echo ""
echo "现在 git push 会同时推送到两个平台"
echo "⚠️  记得先在两个平台创建同名仓库"
