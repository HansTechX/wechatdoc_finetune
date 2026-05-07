#!/usr/bin/env python3
"""打包脚本 - 生成遵循 .gitignore 的部署包，支持推送到远程服务器"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """运行 shell 命令"""
    return subprocess.run(cmd, capture_output=capture, text=True)


def confirm_action(message: str) -> bool:
    """确认操作"""
    while True:
        reply = input(f"{message} (y/N) ").strip().lower()
        if reply == "y":
            return True
        elif reply in ("n", ""):
            return False


def main():
    parser = argparse.ArgumentParser(
        description="项目打包脚本，支持推送到远程服务器"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="打包后推送到远程服务器",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="ali-pai-dsw",
        help="SSH 别名或远程主机地址（默认: ali-pai-dsw）",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="/mnt/wechat_finetune",
        help="远程目标目录路径（默认: /mnt/wechat_finetune）",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"wechatdoc_finetune_{timestamp}.tar.gz"
    package_path = project_root / package_name

    print("=" * 60)
    print("  项目打包脚本")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"输出文件: {package_name}")
    if args.push:
        print(f"推送目标: {args.remote}:{args.remote_path}")
    print()

    # 检查是否在 git 仓库中
    result = run_cmd(["git", "rev-parse", "--git-dir"], capture=False)
    if result.returncode != 0:
        print("错误: 当前目录不是 git 仓库")
        sys.exit(1)

    # 检查是否有未提交的更改
    result = run_cmd(["git", "diff-index", "--quiet", "HEAD", "--"])
    if result.returncode != 0:
        print("警告: 存在未提交的更改")
        print("建议先提交更改，或按 Ctrl+C 取消")
        print()
        if not confirm_action("是否继续打包?"):
            print("已取消")
            sys.exit(1)

    # 打包
    print("正在打包...")
    result = run_cmd(
        ["git", "archive", "--format=tar.gz", f"--output={package_path}", "HEAD"],
        capture=False,
    )

    if result.returncode != 0:
        print("✗ 打包失败")
        sys.exit(1)

    size = package_path.stat().st_size
    size_mb = size / 1024 / 1024
    print()
    print(f"✓ 打包成功: {package_name} ({size_mb:.2f} MB)")

    # 推送到远程服务器
    if args.push:
        print()
        print("-" * 60)
        print("准备推送到远程服务器")
        print("-" * 60)
        print(f"远程主机: {args.remote}")
        print(f"目标目录: {args.remote_path}")
        print(f"压缩文件: {package_name}")
        print()
        print("⚠️  警告：以下操作将会：")
        print(f"  1. 清空远程目录 {args.remote_path} 下的所有文件")
        print(f"  2. 上传并解压 {package_name} 到该目录")
        print()

        if not confirm_action("确认执行推送操作?"):
            print("已取消推送")
            sys.exit(0)

        # 上传文件
        print("正在上传文件...")
        result = run_cmd(
            ["scp", str(package_path), f"{args.remote}:{args.remote_path}/"],
            capture=False,
        )
        if result.returncode != 0:
            print("✗ 上传失败")
            sys.exit(1)
        print("✓ 上传成功")

        # 远程执行：创建目录（如不存在）、清空目录（保留压缩包）和解压
        remote_package_path = f"{args.remote_path}/{package_name}"
        print("正在远程操作...")

        remote_commands = [
            # 创建目录（如不存在）
            f"mkdir -p {args.remote_path}",
            # 进入目标目录
            f"cd {args.remote_path}",
            # 清空当前目录下的所有文件和文件夹（除了刚上传的压缩包）
            f"find . -mindepth 1 -maxdepth 1 ! -name '{package_name}' -exec rm -rf {{}} +",
            # 解压
            f"tar -xzf {package_name}",
            # 删除压缩包
            f"rm {package_name}",
        ]

        remote_cmd = " && ".join(remote_commands)
        result = run_cmd(["ssh", args.remote, remote_cmd], capture=False)

        if result.returncode != 0:
            print("✗ 远程操作失败")
            sys.exit(1)

        print("✓ 远程操作成功")
        print()
        print("=" * 60)
        print(f"✓ 部署完成！项目已推送到 {args.remote}:{args.remote_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
