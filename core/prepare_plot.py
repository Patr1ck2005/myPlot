from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union
from datetime import datetime
import pickle
import gzip
import re
import os

DataClass = Literal["Spectrum", "Eigensolution", "Custom"]

def _safe_str(value: Any) -> str:
    """
    将任意值转为适合文件名的短字符串:
    - 去空白
    - 非 [A-Za-z0-9._-] 的字符替换为 '-'
    - 去掉首尾的 .-_
    """
    s = str(value)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w.\-]+", "-", s)
    return s.strip("._-") or "NA"

def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """
    原子写入：先写到同目录的临时文件，再 replace 到目标文件。
    """
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)

def prepare_plot_data(
    coords: Any,
    data_class: DataClass = "Eigensolution",
    dataset_list: Optional[Sequence[Any]] = None,
    fixed_params: Optional[Mapping[str, Any]] = None,
    save_dir: Union[str, os.PathLike[str]] = "./rsl",
    save_manual_name: Optional[str] = None,
    *,
    compress: bool = False,
    copy_path_to_clipboard: bool = False,
) -> str:
    """
    生成“纯净绘图数据”，保存到带时间戳的子目录，并在当前目录保存一份临时副本。
    参数:
        coords:        坐标/网格等原始坐标数据
        data_class:    数据类别，"Spectrum" | "Eigensolution" | "Custom"
        dataset_list:  数据集序列（None 会被转为 []）
        fixed_params:  会体现在文件名中的固定参数字典（None 会被转为 {}）
        save_dir:      主保存目录（默认 ./rsl）
        save_manual_name: 若提供则强制作为文件名（自动清洗）
        compress:      True 时使用 gzip 压缩（扩展名 .pkl.gz）
        copy_path_to_clipboard: True 时尝试把“主文件绝对路径”复制到剪贴板
    返回:
        str: 主保存文件的绝对路径
    """
    dataset_list = list(dataset_list) if dataset_list is not None else []
    fixed_params = dict(fixed_params) if fixed_params is not None else {}

    # 组装规范结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_data = {
        "coords": coords,
        "data_list": dataset_list,
        "metadata": {
            "fixed_params": fixed_params,
            "timestamp": timestamp,
            "version": "2.1",
            "data_class": data_class,
        },
    }

    # 目录与文件名
    save_root = Path(save_dir)
    data_dir = (save_root / timestamp)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 基于 fixed_params 生成参数段
    if save_manual_name:
        base_name = _safe_str(save_manual_name)
        param_part = base_name
    else:
        if fixed_params:
            kv_parts = [f"{_safe_str(k)}-{_safe_str(v)}" for k, v in sorted(fixed_params.items(), key=lambda kv: str(kv[0]))]
            param_part = "_".join(kv_parts)
        else:
            param_part = "default"

    # 构建最终文件名并做长度保护（Windows 等平台对路径敏感）
    suffix = ".pkl.gz" if compress else ".pkl"
    filename = f"plot_data_{param_part}{suffix}"
    if len(filename) > 200:
        # 截断 param_part，保留前 180 字符（为前缀与后缀留空间）
        param_part = param_part[:180]
        filename = f"plot_data_{param_part}{suffix}"

    file_path = data_dir / filename

    # 序列化（可选 gzip 压缩）
    if compress:
        payload = gzip.compress(pickle.dumps(plot_data, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False))
    else:
        payload = pickle.dumps(plot_data, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)

    # 原子写入主文件
    _atomic_write_bytes(file_path, payload)

    # 控制台提示
    abs_dir = str(data_dir.resolve())
    print(f"纯净绘图数据已保存为：{str(file_path.resolve())} 🎉")
    print(f"文件夹绝对路径：{abs_dir}")

    # 尝试把主文件绝对路径复制到剪贴板（可选）
    if copy_path_to_clipboard:
        try:
            import pyperclip  # 可选依赖
            pyperclip.copy(str(file_path.resolve()))
            print("主文件路径已复制到剪贴板 ✅")
        except Exception:
            # 未安装或环境不支持时，静默跳过
            pass

    # 在当前目录再保存一份未压缩的临时副本，方便快速访问（与原逻辑一致）
    temp_path = Path("./temp_plot_data.pkl").resolve()
    temp_payload = pickle.dumps(plot_data, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
    _atomic_write_bytes(temp_path, temp_payload)
    print(f"临时数据已保存为：{str(temp_path)} 🎉")

    return str(file_path.resolve())
