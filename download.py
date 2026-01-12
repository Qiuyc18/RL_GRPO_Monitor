#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

# --- 文档说明 ---
DESCRIPTION_TEXT = dedent("""\
    [常用命令示例]
    1. 搜索模型:
       python download.py search Qwen --limit 20
    2. 下载模型 (默认下载到 ./models/，没有会创建):
       python download.py download Qwen/Qwen2.5-1.5B-Instruct
    3. 指定目录下载 (推荐部署流程):
       # 下载到临时目录
       python download.py download Qwen/Qwen2.5-1.5B-Instruct --local-dir tmp
       # 移动到公共目录
       sudo mv tmp /etc/moreh/checkpoint/Qwen/Qwen2.5-1.5B-Instruct
       # (可选) 创建只读软链接到当前目录，请确保 ./models 目录存在
       ln -s /etc/moreh/checkpoint/Qwen ./models/
    """)


def _read_env_token(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"HUGGINGFACE_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"}:
            return value or None
    return None


def _get_token(token: str | None = None) -> str | None:
    if token:
        return token
    repo_root = Path(__file__).resolve().parents[1]
    token = _read_env_token(repo_root / ".env")
    if token:
        return token
    return (
        os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Missing dependency: huggingface_hub. Please install it first.")
        return 1

    api = HfApi(token=_get_token())
    results = api.list_models(search=args.query, limit=args.limit)
    for model in results:
        print(model.modelId)  # pyright: ignore[reportAttributeAccessIssue]
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("错误: 缺少依赖库 huggingface_hub。请先运行: pip install huggingface_hub")
        return 1

    token = _get_token(args.token)

    # 路径处理逻辑
    if args.local_dir:
        local_dir = str(Path(args.local_dir))
    else:
        # 默认行为：下载到当前目录下的 models/Author/ModelName
        local_dir = str(Path("models") / args.model_id)

    print(f"准备下载: {args.model_id}")
    print(f"目标路径: {local_dir}")
    if token:
        print("状态: 使用已认证 Token")
    else:
        print("状态: 未检测到 Token，尝试匿名下载...")

    try:
        download_kwargs = {
            "repo_id": args.model_id,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "token": token,
            "max_workers": 8,
        }
        if args.revision is not None:
            download_kwargs["revision"] = args.revision
        snapshot_download(**download_kwargs) 
        print(f"\n[成功] 模型已下载至: {os.path.abspath(local_dir)}")
        return 0
    except Exception as e:
        print(f"\n[失败] 下载出错: {e}")
        if "401" in str(e) or "403" in str(e):
            print("提示: 此模型可能需要权限认证。请提供有效的 HF_TOKEN。")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 全局参数
    parser.add_argument(
        "--token",
        help="HuggingFace Token (可选，也可以通过环境变量设置）",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search 命令
    search_parser = subparsers.add_parser("search", help="按关键词搜索模型")
    search_parser.add_argument("query", help="搜索关键词, 例如: Qwen")
    search_parser.add_argument("--limit", type=int, default=20, help="显示结果数量限制")
    search_parser.set_defaults(func=_cmd_search)

    # Download 命令
    download_parser = subparsers.add_parser("download", help="下载指定模型")
    download_parser.add_argument(
        "model_id", help="模型 ID, 例如: Qwen/Qwen2.5-1.5B-Instruct"
    )
    download_parser.add_argument(
        "--revision", default=None, help="模型分支或 Commit ID"
    )
    download_parser.add_argument("--local-dir", default=None, help="下载目标路径")
    download_parser.set_defaults(func=_cmd_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n操作已取消。")
        sys.exit(1)
