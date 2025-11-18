import threading
import re

import numpy as np
from PIL import Image, ImageOps, ImageFilter


class FormulaRecognizer:
    """Thin wrapper around pix2tex LatexOCR with lazy loading.

    If pix2tex is not installed, raises an informative ImportError.
    """

    _lock = threading.Lock()
    _model = None

    @classmethod
    def _ensure_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    try:
                        from pix2tex.cli import LatexOCR  # type: ignore
                    except Exception as e:
                        raise ImportError(
                            "未安装公式识别引擎 pix2tex。请先运行:\n"
                            "  pip install pix2tex einops transformers\n"
                            "注意: 首次使用需要安装 PyTorch，建议参考 https://pytorch.org/get-started/locally/ \n"
                            f"导入错误: {e}"
                        )
                    cls._model = LatexOCR()

    @staticmethod
    def preprocess_image(
        img: Image.Image,
        *,
        min_side: int = 448,
        max_side: int = 1024,
    ) -> Image.Image:
        """改进的图像预处理，针对手写公式优化。"""
        # 转换为灰度图
        gray = ImageOps.grayscale(img)
        
        # 增强对比度
        gray = ImageOps.autocontrast(gray, cutoff=2)
        arr = np.array(gray, dtype=np.uint8)

        # 判断是否需要反相（统一为白底黑字）
        mean_val = arr.mean()
        if mean_val < 110:
            gray = ImageOps.invert(gray)
            arr = 255 - arr

        # 改进的去噪：先高斯模糊去噪，再中值滤波
        gray = gray.filter(ImageFilter.GaussianBlur(radius=0.5))
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        arr = np.array(gray, dtype=np.uint8)

        # 改进的自动裁剪：使用更精确的阈值
        # 使用自适应阈值找到笔迹区域
        threshold = max(40, min(200, int(np.percentile(arr, 70))))
        mask = arr < threshold
        
        # 如果mask太小，使用更宽松的阈值
        if mask.sum() < arr.size * 0.01:
            threshold = max(30, int(np.percentile(arr, 60)))
            mask = arr < threshold
        
        coords = np.argwhere(mask)
        if coords.size:
            (ymin, xmin), (ymax, xmax) = coords.min(axis=0), coords.max(axis=0)
            # 增加边距，确保不丢失信息
            pad = max(16, int(0.1 * max(ymax - ymin, xmax - xmin)))
            xmin = max(int(xmin) - pad, 0)
            ymin = max(int(ymin) - pad, 0)
            xmax = min(int(xmax) + pad, gray.width - 1)
            ymax = min(int(ymax) + pad, gray.height - 1)
            gray = gray.crop((xmin, ymin, xmax + 1, ymax + 1))
            arr = np.array(gray, dtype=np.uint8)

        # 改进的二值化：使用自适应阈值
        thresh = max(40, min(220, int(np.percentile(arr, 70))))
        binary_arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
        binary = Image.fromarray(binary_arr, mode='L')
        
        # 形态学操作：先膨胀增强细笔画，再中值滤波去噪
        binary = binary.filter(ImageFilter.MaxFilter(size=3))
        binary = binary.filter(ImageFilter.MedianFilter(size=3))
        
        # 轻微锐化
        binary = binary.filter(ImageFilter.SHARPEN)
        
        # 添加白色边框
        binary = ImageOps.expand(binary, border=16, fill=255)

        # 缩放处理
        w, h = binary.size
        longest = max(w, h)
        scale = 1.0
        if longest < min_side:
            scale = min_side / max(longest, 1)
        elif longest > max_side:
            scale = max_side / longest
        if scale != 1.0:
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except AttributeError:  # Pillow<9.1
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            binary = binary.resize((new_w, new_h), resample=resample)

        # 最终增强对比度
        binary = ImageOps.autocontrast(binary, cutoff=2)
        
        return binary.convert('RGB')

    @classmethod
    @staticmethod
    def _candidate_temperatures():
        # lower sampling temp yields stabler results; higher temp keeps flexibility
        return (0.18, 0.22, 0.3)

    @staticmethod
    def _cleanup_latex(latex: str) -> str:
        if not latex:
            return ""
        # collapse whitespace but keep intentional spacing around commands
        latex = re.sub(r"\s+", " ", latex).strip()
        # if the whole expression is duplicated multiple times, keep only one copy
        repeat_pattern = re.compile(r"^(.+?)(?:\s*\1){1,}$")
        match = repeat_pattern.match(latex)
        if match:
            latex = match.group(1).strip()
        return latex

    @staticmethod
    def _repeat_penalty(latex: str) -> float:
        tokens = re.findall(r"(\\[A-Za-z]+|\\.|[{}()\[\]]|\d+|[^\\{}\s])", latex)
        if len(tokens) < 2:
            return 0.0
        repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        return repeats / max(len(tokens) - 1, 1)

    @staticmethod
    def _score_candidate(latex: str) -> float:
        if not latex:
            return float("-inf")
        normalized = latex.replace(" ", "")
        brace_gap = abs(latex.count("{") - latex.count("}"))
        env_gap = abs(latex.count(r"\begin") - latex.count(r"\end"))
        repeat_pen = FormulaRecognizer._repeat_penalty(latex)
        length_pen = max(0, len(normalized) - 200) * 0.004
        penalty = repeat_pen * 8.0 + brace_gap * 0.5 + env_gap * 1.5 + length_pen
        return 1.0 - penalty

    @classmethod
    def recognize_latex(cls, img: Image.Image, *, preprocessed: bool = False) -> str:
        cls._ensure_model()
        assert cls._model is not None

        proc = img if preprocessed else cls.preprocess_image(img)
        candidates = []
        with cls._lock:
            # mutate sampling temperature sequentially, collect multiple attempts
            original_temp = getattr(cls._model.args, "temperature", None)  # type: ignore[attr-defined]
            for temp in cls._candidate_temperatures():
                try:
                    cls._model.args.temperature = temp  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                try:
                    latex = cls._model(proc)
                except Exception:
                    continue
                cleaned = cls._cleanup_latex(latex)
                candidates.append((cls._score_candidate(cleaned), cleaned))
            if original_temp is not None:
                try:
                    cls._model.args.temperature = original_temp  # type: ignore[attr-defined]
                except AttributeError:
                    pass

        if not candidates:
            raise RuntimeError("公式识别失败：未获得有效输出")

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
