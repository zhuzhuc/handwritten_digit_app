import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别应用")
        self.root.minsize(1180, 720)
        self.palette = {
            "bg": "#0f172a",
            "hero_bg": "#111c3a",
            "card": "#1f2a44",
            "border": "#2d3a5c",
            "text": "#f8fafc",
            "muted": "#96a3c7",
            "accent": "#22d3ee",
            "accent_hover": "#0ea5e9",
        }
        self.root.configure(bg=self.palette["bg"])
        self.canvas_width = 640
        self.canvas_height = 480
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = self._load_recognition_model()
            print(f"模型加载成功！(device={self.device})")
        except Exception as exc:
            messagebox.showerror(
                "错误",
                "无法加载 PyTorch 模型文件！\n"
                "请先运行 train_model.py 并确保生成 model_scripted.pt。\n"
                f"详细信息：{exc}"
            )
            root.destroy()
            return

        self.status_var = tk.StringVar(value="就绪")
        self.history_limit = 20
        self.digit_history_widget = None
        self.brush_radius = tk.DoubleVar(value=12.0)
        self.last_point = None
        self.topk_labels = []
        self.topk_bars = []
        self._setup_style()
        self._build_layout()
        self.center_window()

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            if self.root.tk.call("tk", "windowingsystem") == "aqua":
                style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=self.palette["bg"])
        style.configure("App.TFrame", background=self.palette["bg"])
        style.configure("Hero.TFrame", background=self.palette["hero_bg"])
        style.configure("HeroTitle.TLabel", background=self.palette["hero_bg"], foreground=self.palette["text"], font=("PingFang SC", 24, "bold"))
        style.configure("HeroSubtitle.TLabel", background=self.palette["hero_bg"], foreground=self.palette["muted"], font=("PingFang SC", 11))
        style.configure("Card.TFrame", background=self.palette["card"], borderwidth=0, relief="flat")
        style.configure("CardTitle.TLabel", background=self.palette["card"], foreground=self.palette["text"], font=("PingFang SC", 13, "bold"))
        style.configure("CardBody.TLabel", background=self.palette["card"], foreground=self.palette["muted"], font=("PingFang SC", 10))
        style.configure("ResultValue.TLabel", background=self.palette["card"], foreground=self.palette["text"], font=("Futura", 34, "bold"))
        style.configure("ResultHint.TLabel", background=self.palette["card"], foreground=self.palette["muted"], font=("PingFang SC", 11))
        style.configure("Muted.TLabel", background=self.palette["card"], foreground=self.palette["muted"])
        style.configure("Statusbar.TLabel", anchor="w", background=self.palette["bg"], foreground=self.palette["muted"])
        style.configure("Accent.TButton", padding=(14, 10), font=("PingFang SC", 12, "bold"), foreground=self.palette["hero_bg"], background=self.palette["accent"], relief="flat")
        style.map(
            "Accent.TButton",
            background=[("active", self.palette["accent_hover"]), ("pressed", self.palette["accent_hover"]), ("disabled", "#1f3b54")],
        )
        style.configure(
            "Secondary.TButton",
            padding=(12, 10),
            background=self.palette["border"],
            foreground=self.palette["text"],
            relief="flat",
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#3b4c73"), ("pressed", "#3b4c73")],
            foreground=[("disabled", "#5f6a89")],
        )
        style.configure("Card.TLabelframe", background=self.palette["card"], bordercolor=self.palette["border"])
        style.configure("Card.TLabelframe.Label", background=self.palette["card"], foreground=self.palette["text"], font=("PingFang SC", 11, "bold"))
        style.configure("Confidence.Horizontal.TProgressbar", troughcolor=self.palette["border"], bordercolor=self.palette["border"], background=self.palette["accent"], lightcolor=self.palette["accent"], darkcolor=self.palette["accent"])

    def _build_layout(self):
        hero = ttk.Frame(self.root, padding=(24, 20, 24, 10), style="Hero.TFrame")
        hero.pack(fill=tk.X)
        ttk.Label(hero, text="手写数字识别工作台", style="HeroTitle.TLabel").pack(anchor="w")
        # ttk.Label(hero, text="更聪明的识别模型 + 焕然一新的界面", style="HeroSubtitle.TLabel").pack(anchor="w", pady=(6, 0))

        self.main_frame = ttk.Frame(self.root, padding=(24, 10, 24, 20), style="App.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=3, uniform="main")
        self.main_frame.columnconfigure(1, weight=2, uniform="main")
        self.main_frame.rowconfigure(0, weight=1)

        canvas_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=22)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        canvas_card.columnconfigure(0, weight=1)
        ttk.Label(canvas_card, text="手写画布", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            canvas_card,
            text="用鼠标或触控板写下 1-5 位数字，保持线条清晰顺滑。",
            style="CardBody.TLabel",
            wraplength=380,
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        self.canvas = tk.Canvas(
            canvas_card,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#fefefe",
            highlightthickness=2,
            highlightbackground=self.palette["border"],
        )
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.canvas.bind("<Leave>", self.end_draw)

        # 离屏绘图，用于稳定的图像获取
        self.sketch = Image.new('L', (self.canvas_width, self.canvas_height), color=255)
        self.drawer = ImageDraw.Draw(self.sketch)

        controls = ttk.Frame(self.main_frame, style="App.TFrame")
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        result_card = ttk.Frame(controls, style="Card.TFrame", padding=18)
        result_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        result_card.columnconfigure(0, weight=1)
        ttk.Label(result_card, text="识别结果", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.label_result = ttk.Label(result_card, text="等待输入", style="ResultValue.TLabel", anchor="w")
        self.label_result.grid(row=1, column=0, sticky="w", pady=(6, 4))
        self.result_detail = ttk.Label(
            result_card,
            text="在画布上书写数字或导入图片以开始识别。",
            style="ResultHint.TLabel",
            wraplength=280,
            justify="left",
        )
        self.result_detail.grid(row=2, column=0, sticky="w")
        ttk.Separator(result_card, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky="ew", pady=(10, 8))
        topk_frame = ttk.Frame(result_card, style="Card.TFrame")
        topk_frame.grid(row=4, column=0, sticky="ew")
        ttk.Label(topk_frame, text="Top-3 候选", style="CardBody.TLabel").grid(row=0, column=0, sticky="w")
        for idx in range(3):
            row = idx * 2 + 1
            label = ttk.Label(topk_frame, text=f"No.{idx+1} — 等待", style="CardBody.TLabel", anchor="w")
            label.grid(row=row, column=0, sticky="ew", pady=(4 if idx == 0 else 2, 0))
            bar = ttk.Progressbar(
                topk_frame,
                style="Confidence.Horizontal.TProgressbar",
                maximum=100,
                length=260,
            )
            bar.grid(row=row + 1, column=0, sticky="ew", pady=(0, 4))
            bar["value"] = 0
            self.topk_labels.append(label)
            self.topk_bars.append(bar)

        instructions_card = ttk.Frame(controls, style="Card.TFrame", padding=16)
        instructions_card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(instructions_card, text="使用提示", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        tips = (
            "• 建议用较粗的笔画并保持节奏顺滑\n"
            "• 多位数字保持合理间距\n"
            "• 如果识别异常，可尝试清空后重写"
        )
        ttk.Label(
            instructions_card, text=tips, style="CardBody.TLabel", justify="left", wraplength=280
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        brush_frame = ttk.LabelFrame(controls, text="画笔设置", style="Card.TLabelframe", padding=10)
        brush_frame.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        brush_frame.columnconfigure(0, weight=1)
        brush_frame.columnconfigure(1, weight=1)

        ttk.Label(brush_frame, text="笔刷直径 (px)", style="CardBody.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4)
        )
        brush_scale = ttk.Scale(
            brush_frame,
            from_=6,
            to=28,
            orient=tk.HORIZONTAL,
            variable=self.brush_radius,
            command=self._update_brush_label,
        )
        brush_scale.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5)
        self.brush_value_label = ttk.Label(brush_frame, text="", width=12, style="CardBody.TLabel")
        self.brush_value_label.grid(row=2, column=0, sticky="w", padx=5, pady=(4, 0))
        ttk.Button(brush_frame, text="清空画布", style="Secondary.TButton", command=self.clear_canvas).grid(
            row=2, column=1, sticky="e", padx=5, pady=(2, 2)
        )
        self._update_brush_label()

        action_frame = ttk.Frame(controls, style="Card.TFrame", padding=16)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        action_frame.columnconfigure((0, 1), weight=1)
        ttk.Button(action_frame, text="识别画布", style="Accent.TButton", command=self.recognize).grid(
            row=0, column=0, padx=(0, 8), pady=4, sticky="ew"
        )
        ttk.Button(action_frame, text="识别图片", style="Secondary.TButton", command=self.recognize_image).grid(
            row=0, column=1, padx=(8, 0), pady=4, sticky="ew"
        )

        history_card = ttk.Frame(controls, style="Card.TFrame", padding=16)
        history_card.grid(row=4, column=0, sticky="nsew")
        history_card.columnconfigure(0, weight=1)
        history_card.rowconfigure(1, weight=1)
        ttk.Label(history_card, text="识别历史", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.digit_history_widget = self._create_history_panel(history_card)

        ttk.Label(controls, textvariable=self.status_var, style="Statusbar.TLabel").grid(
            row=5, column=0, sticky="ew", pady=(10, 0)
        )

        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, style="Statusbar.TLabel", padding=(16, 6)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _create_history_panel(self, parent):
        panel = ttk.Frame(parent, style="Card.TFrame")
        panel.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        text_widget = tk.Text(
            panel,
            height=8,
            width=30,
            state="disabled",
            wrap="word",
            relief="flat",
            bg=self.palette["card"],
            fg=self.palette["text"],
        )
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(2, 0))
        ttk.Button(
            parent, text="清空历史", style="Secondary.TButton", command=lambda w=text_widget: self._clear_history(w)
        ).grid(row=2, column=0, sticky="e", pady=(6, 0))
        return text_widget
        
    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def _build_fallback_model(self):
        class ConvClassifier(nn.Module):
            def __init__(self, hidden_dim: int = 128):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.05),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, hidden_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 10),
                )

                def forward(self, x):
                    feats = self.features(x)
                    logits = self.classifier(feats)
                    return F.log_softmax(logits, dim=1)

        return ConvClassifier()

    def _load_recognition_model(self):
        script_path = Path("model_scripted.pt")
        state_path = Path("model_state.pt")
        if script_path.exists():
            model = torch.jit.load(script_path, map_location=self.device)
        elif state_path.exists():
            model = self._build_fallback_model()
            state_dict = torch.load(state_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError("未找到 model_scripted.pt 或 model_state.pt")
        model.to(self.device)
        model.eval()
        return model

    def _current_brush_radius(self) -> int:
        return max(4, int(round(self.brush_radius.get())))

    def _update_brush_label(self, *_):
        if hasattr(self, "brush_value_label"):
            diameter = self._current_brush_radius() * 2
            self.brush_value_label.config(text=f"{diameter} px")

    def start_draw(self, event):
        """记录按下位置并绘制初始笔迹。"""
        self.last_point = (event.x, event.y)
        self._draw_dot(event.x, event.y)

    def draw(self, event):
        """跟随鼠标绘制平滑的曲线。"""
        if self.last_point is None:
            self.start_draw(event)
            return
        x0, y0 = self.last_point
        x1, y1 = event.x, event.y
        width = self._current_brush_radius() * 2
        self.canvas.create_line(x0, y0, x1, y1, fill="black", width=width, capstyle=tk.ROUND, smooth=True)
        self.drawer.line((x0, y0, x1, y1), fill=0, width=width)
        self.last_point = (x1, y1)

    def end_draw(self, _event=None):
        """结束当前笔画。"""
        self.last_point = None

    def _draw_dot(self, x: int, y: int):
        r = self._current_brush_radius()
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="")
        self.drawer.ellipse((x-r, y-r, x+r, y+r), fill=0)

    def clear_canvas(self):
        """清空画布"""
        self.canvas.delete("all")
        self.label_result.config(text="等待输入")
        self.result_detail.config(text="画布已清空，写下新的数字即可再次识别。")
        self._update_topk_panel([])
        self._set_status("画布已清空")
        self.last_point = None
        # 重置离屏图像
        self.sketch = Image.new('L', self.sketch.size, color=255)
        self.drawer = ImageDraw.Draw(self.sketch)
    
    def _append_history(self, widget: tk.Text, text: str):
        if widget is None:
            return
        widget.configure(state="normal")
        widget.insert("end", text + "\n")
        widget.see("end")
        content = widget.get("1.0", "end-1c").splitlines()
        if len(content) > self.history_limit:
            trimmed = "\n".join(content[-self.history_limit:])
            widget.delete("1.0", "end")
            widget.insert("1.0", trimmed + "\n")
        widget.configure(state="disabled")

    def _clear_history(self, widget: tk.Text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")

    def _append_digit_history(self, source: str, digits: str, confidences):
        conf_text = ", ".join(f"{d}:{c:.0%}" for d, c in confidences)
        line = f"[{source}] {digits} | {conf_text}"
        self._append_history(self.digit_history_widget, line)

    def _prepare_digit_segments(self, img: Image.Image):
        """从原始图像中提取一个或多个数字的裁剪图像（改进版）。"""
        # 转换为灰度图
        gray = img.convert('L')
        np_gray = np.array(gray, dtype=np.uint8)
        
        # 判断是否需要反相（确保是白底黑字）
        mean_val = np_gray.mean()
        if mean_val > 127:
            # 背景较亮，需要反相
            proc = ImageOps.invert(gray)
            np_gray = 255 - np_gray
        else:
            proc = gray
        
        # 轻微去噪，但不要过度模糊
        proc = proc.filter(ImageFilter.MedianFilter(size=3))
        
        # 改进的二值化：使用改进的Otsu方法
        arr = np.array(proc, dtype=np.uint8)
        
        # 计算自适应阈值（使用Otsu方法）
        hist, bins = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum()
        
        # 找到最佳阈值
        cumsum = np.cumsum(hist)
        cummean = np.cumsum(hist * np.arange(256))
        total_mean = cummean[-1]
        
        best_thresh = 128
        best_var = 0
        for t in range(1, 255):
            w0 = cumsum[t]
            w1 = 1 - w0
            if w0 == 0 or w1 == 0:
                continue
            m0 = cummean[t] / w0
            m1 = (total_mean - cummean[t]) / w1
            var_between = w0 * w1 * (m0 - m1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = t
        
        # 使用改进的阈值策略：更保守，避免丢失细节
        # 对于手写数字，使用稍低的阈值以保留细笔画
        adaptive_thresh = max(40, min(180, int(best_thresh * 0.75)))
        mask = arr > adaptive_thresh
        
        # 如果mask为空或太小，使用更宽松的阈值
        if mask.sum() == 0 or mask.sum() < arr.size * 0.001:
            adaptive_thresh = max(30, int(np.percentile(arr, 55)))
            mask = arr > adaptive_thresh
        
        coords = np.argwhere(mask)
        if coords.size == 0:
            return []

        # 裁剪数字区域，增加边距
        (ymin, xmin), (ymax, xmax) = coords.min(axis=0), coords.max(axis=0)
        pad = max(12, int(0.1 * max(ymax - ymin, xmax - xmin)))
        xmin = max(int(xmin) - pad, 0)
        ymin = max(int(ymin) - pad, 0)
        xmax = min(int(xmax) + pad, proc.width - 1)
        ymax = min(int(ymax) + pad, proc.height - 1)
        digit_img = proc.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        # 增强对比度和清晰度
        digit_img = ImageOps.autocontrast(digit_img, cutoff=1.5)
        # 轻微锐化，但不要过度
        digit_img = digit_img.filter(ImageFilter.SHARPEN)
        # 轻微膨胀以连接断裂的笔画
        digit_img = digit_img.filter(ImageFilter.MaxFilter(size=3))

        # 分割多个数字 - 改进的分割逻辑
        seg_arr = np.array(digit_img, dtype=np.uint8)
        # 使用更合适的阈值来分割
        seg_thresh = max(40, int(np.percentile(seg_arr, 60)))
        seg_mask = seg_arr > seg_thresh
        seg_mask = self._denoise_digit_mask(seg_mask)
        
        # 先尝试连通域分割
        component_boxes = self._connected_components(seg_mask)
        filtered_boxes = self._filter_digit_boxes(component_boxes, seg_mask.shape)
        crops = []
        
        if filtered_boxes and len(filtered_boxes) > 0:
            # 如果检测到多个连通域，使用连通域分割
            merged_boxes = self._merge_digit_boxes(filtered_boxes, seg_mask.shape)
            # 限制最多5个数字
            if len(merged_boxes) > 5:
                # 如果超过5个，按宽度排序，取最大的5个
                merged_boxes = sorted(merged_boxes, key=lambda b: (b[1]-b[0])*(b[3]-b[2]), reverse=True)[:5]
                merged_boxes = sorted(merged_boxes, key=lambda b: b[0])  # 按x坐标排序
            
            for xmin, xmax, ymin, ymax in merged_boxes:
                # 增加边距以确保不丢失信息
                extra_pad = max(4, int(0.05 * max(xmax - xmin, ymax - ymin)))
                crop = digit_img.crop(
                    (
                        max(0, xmin - extra_pad),
                        max(0, ymin - extra_pad),
                        min(digit_img.width, xmax + extra_pad),
                        min(digit_img.height, ymax + extra_pad),
                    )
                )
                crop = ImageOps.autocontrast(crop, cutoff=1.5)
                crops.append(crop)
        else:
            # 如果连通域分割失败，使用列投影分割
            segments = self._split_digit_columns(seg_mask)
            # 限制最多5个数字
            if len(segments) > 5:
                segments = segments[:5]
            
            for start, end in sorted(segments, key=lambda span: span[0]):
                if end - start <= 0:
                    continue
                # 增加垂直边距
                v_pad = max(4, int(0.05 * digit_img.height))
                crop = digit_img.crop((
                    max(0, start - 2), 
                    max(0, -v_pad), 
                    min(digit_img.width, end + 2), 
                    min(digit_img.height, digit_img.height + v_pad)
                ))
                crop = ImageOps.autocontrast(crop, cutoff=1.5)
                crops.append(crop)
        
        # 如果还是没有分割出任何数字，返回整个图像作为单个数字
        if not crops:
            crops.append(ImageOps.autocontrast(digit_img, cutoff=1.5))
        
        return crops

    def _denoise_digit_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """简单的形态学闭运算，抑制小噪声并连接断裂笔画。"""
        pil_mask = Image.fromarray((binary_mask.astype(np.uint8)) * 255, mode='L')
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=5))
        pil_mask = pil_mask.filter(ImageFilter.MinFilter(size=3))
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=3))
        return (np.array(pil_mask, dtype=np.uint8) > 0)

    def _filter_digit_boxes(self, boxes, mask_shape):
        """根据高度/宽度过滤显然是噪声的连通域。"""
        height, width = mask_shape
        # 放宽过滤条件，避免误删有效数字
        min_height = max(8, int(height * 0.15))
        min_width = max(4, int(width * 0.01))
        filtered = []
        for xmin, xmax, ymin, ymax in boxes:
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            # 放宽条件：只要高度或宽度满足一个即可
            if bh >= min_height or bw >= min_width:
                # 但面积不能太小
                area = bw * bh
                min_area = max(20, int(height * width * 0.01))
                if area >= min_area:
                    filtered.append([xmin, xmax, ymin, ymax])
        return filtered

    def _vertical_overlap(self, box_a, box_b):
        top = max(box_a[2], box_b[2])
        bottom = min(box_a[3], box_b[3])
        return max(0, bottom - top)

    def _merge_digit_boxes(self, boxes, mask_shape):
        """相邻的候选框如果高度重叠足够大，就归并为单个数字区域。"""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0][:]]
        gap = max(4, int(mask_shape[1] * 0.015))
        for box in boxes[1:]:
            last = merged[-1]
            overlap = self._vertical_overlap(last, box)
            min_height = min(last[3] - last[2], box[3] - box[2])
            if box[0] - last[1] <= gap and overlap >= 0.4 * max(1, min_height):
                last[1] = max(last[1], box[1])
                last[2] = min(last[2], box[2])
                last[3] = max(last[3], box[3])
            else:
                merged.append(box[:])
        return merged

    def _connected_components(self, binary_mask: np.ndarray):
        """连通域分割，避免列投影误拆/误合。"""
        h, w = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        boxes = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y in range(h):
            for x in range(w):
                if not binary_mask[y, x] or visited[y, x]:
                    continue
                queue = deque([(y, x)])
                visited[y, x] = True
                ymin = ymax = y
                xmin = xmax = x
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            ymin = min(ymin, ny)
                            ymax = max(ymax, ny)
                            xmin = min(xmin, nx)
                            xmax = max(xmax, nx)
                boxes.append((xmin, xmax + 1, ymin, ymax + 1))
        boxes.sort(key=lambda b: b[0])
        return boxes

    def _split_digit_columns(self, binary_mask: np.ndarray):
        """基于列投影的改进多数字分割。"""
        col_hist = binary_mask.sum(axis=0)
        height = binary_mask.shape[0]
        # 降低阈值，避免丢失细笔画
        min_foreground = max(1, int(height * 0.03))
        active = col_hist >= min_foreground

        # 动态计算最小宽度，根据图像大小调整
        min_width = max(4, int(binary_mask.shape[1] * 0.06))
        pad = 3
        # 增加gap容忍度，避免把单个数字分割成多个
        gap_tolerance = max(3, int(binary_mask.shape[1] * 0.02))
        segments = []
        start = None
        gap = 0
        
        for idx, flag in enumerate(active):
            if flag:
                if start is None:
                    start = idx
                gap = 0
            else:
                if start is not None:
                    gap += 1
                    if gap > gap_tolerance:
                        end = idx - gap
                        if end - start >= min_width:
                            segments.append((max(0, start - pad), min(binary_mask.shape[1], end + pad)))
                        start = None
                        gap = 0
        if start is not None:
            end = len(active) - 1
            if end - start + 1 >= min_width:
                segments.append((max(0, start - pad), min(len(active), end + 1 + pad)))

        if not segments:
            segments.append((0, binary_mask.shape[1]))
        
        # 限制最多5个数字
        if len(segments) > 5:
            # 按宽度排序，保留最宽的5个
            segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:5]
            segments = sorted(segments, key=lambda s: s[0])  # 按位置排序
        
        return segments

    def _digit_image_to_array(self, digit_img: Image.Image):
        """将单个数字裁剪图像转换为模型输入（改进版）。"""
        w, h = digit_img.size
        if w <= 0 or h <= 0:
            return None
        
        # 改进的缩放：保持宽高比，缩放到20像素（留出边距）
        # 使用稍小的缩放比例，确保数字不会太大
        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        
        # 使用高质量重采样
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        resized = digit_img.resize((new_w, new_h), resample=resample)

        # 创建28x28画布，背景为黑色（0）
        canvas28 = Image.new('L', (28, 28), color=0)
        left = (28 - new_w) // 2
        top = (28 - new_h) // 2
        canvas28.paste(resized, (left, top))

        # 改进的居中算法：使用质心计算
        np_canvas = np.array(canvas28, dtype=np.float32)
        
        # 计算非零像素的质心 - 降低阈值以包含更多像素
        nonzero = np_canvas > 5  # 降低阈值，避免丢失细节
        if nonzero.sum() == 0:
            # 如果没有有效像素，返回None
            return None
        
        y_coords, x_coords = np.where(nonzero)
        total_intensity = np_canvas[nonzero].sum()
        
        if total_intensity > 0:
            # 使用加权质心（考虑像素强度）
            cy = (y_coords * np_canvas[nonzero]).sum() / total_intensity
            cx = (x_coords * np_canvas[nonzero]).sum() / total_intensity
        else:
            # 如果总强度为0，使用几何质心
            cy = y_coords.mean()
            cx = x_coords.mean()
        
        # 计算需要移动的距离（目标中心是14, 14）
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        
        # 限制移动范围，避免移出边界
        shift_x = max(-12, min(12, shift_x))
        shift_y = max(-12, min(12, shift_y))
        
        if shift_x != 0 or shift_y != 0:
            canvas28 = ImageChops.offset(canvas28, shift_x, shift_y)

        # 去倾斜处理
        canvas28 = self._deskew_canvas(canvas28)

        # 最终增强：轻微锐化
        canvas28 = canvas28.filter(ImageFilter.SHARPEN)
        
        # 转换为模型输入格式
        img_array = np.array(canvas28, dtype=np.float32) / 255.0
        
        # 归一化：确保值在[0, 1]范围内
        img_array = np.clip(img_array, 0.0, 1.0)
        
        # 改进的对比度增强：使用更温和的gamma校正
        # 对于手写数字，稍微增强对比度有助于识别
        img_array = np.power(img_array, 0.85)  # gamma校正，增强对比度
        
        # 归一化到[0,1]范围
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min)
        
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

    def _deskew_canvas(self, canvas: Image.Image):
        """根据二阶矩对28x28图像做轻量去倾斜处理。"""
        arr = np.array(canvas, dtype=np.float32)
        mask = arr > 10
        if not mask.any():
            return canvas
        y_coords, x_coords = np.where(mask)
        intensities = arr[mask]
        total = intensities.sum()
        if total <= 0:
            return canvas
        cy = float((y_coords * intensities).sum() / total)
        cx = float((x_coords * intensities).sum() / total)
        mu11 = float(((x_coords - cx) * (y_coords - cy) * intensities).sum() / total)
        mu02 = float((((y_coords - cy) ** 2) * intensities).sum() / total)
        if abs(mu02) < 1e-3:
            return canvas
        skew = mu11 / mu02
        if abs(skew) < 0.01 or abs(skew) > 0.7:
            return canvas
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        transform = (1, skew, -skew * cy, 0, 1, 0)
        return canvas.transform(canvas.size, Image.AFFINE, transform, resample=resample)

    def _digit_tta_variants(self, digit_array: np.ndarray):
        """为单个数字生成轻量 TTA 版本，提高预测稳定性。"""
        base = np.clip(digit_array, 0.0, 1.0)
        image = (base[0, :, :, 0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(image, mode='L')
        variants = [base]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = ImageChops.offset(pil_img, dx, dy)
            shifted_arr = np.array(shifted, dtype=np.float32) / 255.0
            variants.append(shifted_arr.reshape(1, 28, 28, 1))
        softened = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))
        softened_arr = np.array(softened, dtype=np.float32) / 255.0
        variants.append(softened_arr.reshape(1, 28, 28, 1))
        return variants

    def _predict_digits(self, digit_arrays):
        """对每个数字执行 TTA，返回平均后的预测结果列表。"""
        if not digit_arrays:
            return []
        batched = []
        spans = []
        for arr in digit_arrays:
            variants = self._digit_tta_variants(arr)
            batched.extend(variants)
            spans.append(len(variants))
        batch = np.concatenate(batched, axis=0).astype(np.float32)
        # 转换为 NCHW (Torch) 格式，并套用训练同款归一化
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch = (batch - MNIST_MEAN) / MNIST_STD
        batch = np.ascontiguousarray(batch)
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.exp(logits).cpu().numpy()
        outputs = []
        idx = 0
        for span in spans:
            # 使用加权平均，原始图像权重更高
            pred_slice = probs[idx:idx + span]
            # 第一个是原始图像，权重为2，其他为1
            if span == 1:
                outputs.append(pred_slice[0])
            else:
                weights = np.array([2.0] + [1.0] * (span - 1), dtype=np.float32)
                weights = weights / weights.sum()
                weighted_pred = np.average(pred_slice, axis=0, weights=weights)
                outputs.append(weighted_pred)
            idx += span
        return outputs

    def recognize(self):
        """识别画布中的数字"""
        self._set_status("识别数字中...")
        segments = self._prepare_digit_segments(self.sketch)
        batch_arrays = [self._digit_image_to_array(seg) for seg in segments]
        batch_arrays = [arr for arr in batch_arrays if arr is not None]
        if not batch_arrays:
            self.label_result.config(text="未检测到笔迹，请重新书写后再试")
            self._set_status("未检测到笔迹")
            self._update_topk_panel([])
            return

        predictions = self._predict_digits(batch_arrays)
        digits = []
        confidences = []
        for pred in predictions:
            digit_idx = int(np.argmax(pred))
            confidence = float(pred[digit_idx])
            # 如果置信度太低，记录警告但仍然使用预测结果
            if confidence < 0.2:
                self._set_status(f"警告：数字 {digit_idx} 置信度较低 ({confidence:.1%})")
            digits.append(str(digit_idx))
            confidences.append(confidence)

        num_digits = len(digits)
        conf_text = " ".join(f"{d}:{c:.0%}" for d, c in zip(digits, confidences))
        if num_digits == 1:
            self.label_result.config(text=digits[0])
            self.result_detail.config(text=f"单数字置信度：{confidences[0]:.2%}")
        else:
            self.label_result.config(text="".join(digits))
            self.result_detail.config(text=f"{num_digits} 位数字，置信度：{conf_text}")
        self._append_digit_history("画布", "".join(digits), list(zip(digits, confidences)))
        self._update_topk_panel(predictions)
        self._set_status("识别完成")

    def recognize_image(self):
        """从图片文件中识别数字"""
        self._set_status("选择数字图片中...")
        file_path = filedialog.askopenfilename(
            title="选择包含数字的图片（支持1-5位数字）",
            filetypes=[
                ("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")),
                ("All Files", ("*.*",))
            ]
        )
        if not file_path:
            self._set_status("已取消选择")
            return
        try:
            img = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片：{e}")
            self._set_status("打开图片失败")
            return

        segments = self._prepare_digit_segments(img)
        batch_arrays = [self._digit_image_to_array(seg) for seg in segments]
        batch_arrays = [arr for arr in batch_arrays if arr is not None]
        if not batch_arrays:
            self.label_result.config(text="未检测到数字，请确保图片中有清晰的数字（支持1-5位）")
            self._set_status("未检测到数字")
            self._update_topk_panel([])
            return

        predictions = self._predict_digits(batch_arrays)
        digits = []
        confidences = []
        for pred in predictions:
            digit_idx = int(np.argmax(pred))
            confidence = float(pred[digit_idx])
            # 如果置信度太低，记录警告但仍然使用预测结果
            if confidence < 0.2:
                self._set_status(f"警告：数字 {digit_idx} 置信度较低 ({confidence:.1%})")
            digits.append(str(digit_idx))
            confidences.append(confidence)

        num_digits = len(digits)
        conf_text = " ".join(f"{d}:{c:.0%}" for d, c in zip(digits, confidences))
        if num_digits == 1:
            self.label_result.config(text=digits[0])
            self.result_detail.config(text=f"图片识别置信度：{confidences[0]:.2%}")
        else:
            self.label_result.config(text="".join(digits))
            self.result_detail.config(text=f"{num_digits} 位数字，置信度：{conf_text}")
        source = f"图片:{Path(file_path).name}"
        self._append_digit_history(source, "".join(digits), list(zip(digits, confidences)))
        self._update_topk_panel(predictions)
        self._set_status("识别完成")

    def _update_topk_panel(self, prob_list):
        if not self.topk_labels or not self.topk_bars:
            return
        if not prob_list:
            for label, bar in zip(self.topk_labels, self.topk_bars):
                label.config(text="—")
                bar["value"] = 0
            return
        stacked = np.stack(prob_list).mean(axis=0)
        order = np.argsort(stacked)[::-1][:3]
        for idx, (label, bar) in enumerate(zip(self.topk_labels, self.topk_bars)):
            if idx < len(order):
                digit = order[idx]
                prob = float(stacked[digit])
                label.config(text=f"No.{idx+1} · 数字 {digit}  ({prob:.1%})")
                bar["value"] = prob * 100
            else:
                label.config(text="—")
                bar["value"] = 0

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
