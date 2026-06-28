"""
FFmpeg GPU 视频裁剪工具 — 单文件，双击运行
"""
import os, sys, subprocess, threading, platform, re, time, queue
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QFont, QColor, QKeySequence, QShortcut, QPolygonF, QValidator, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QProgressBar,
    QGraphicsView, QGraphicsScene, QComboBox, QStatusBar, QSizePolicy, QSpinBox, QStyle,
    QSplitter, QListWidget, QListWidgetItem,
)
import cv2
import numpy as np

_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) if platform.system() == "Windows" else 0

FFMPEG = None
HAS_CUDA = False
NVENC_ENC = None
for _c in ["ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
    try:
        r = subprocess.run([_c, "-version"], capture_output=True, text=True, errors='replace', timeout=5)
        if r.returncode: continue
        FFMPEG = _c
        hw = subprocess.run([_c, "-hwaccels"], capture_output=True, text=True, errors='replace', timeout=5)
        en = subprocess.run([_c, "-encoders"], capture_output=True, text=True, errors='replace', timeout=5)
        HAS_CUDA = "cuda" in hw.stdout.lower()
        for e in ("av1_nvenc", "hevc_nvenc", "h264_nvenc"):
            if e in en.stdout: NVENC_ENC = e; break
        break
    except Exception:
        pass

ENC_OPTIONS = [("自动选最优", None), ("AV1 (最快)", "av1_nvenc"), ("HEVC", "hevc_nvenc"), ("H.264", "h264_nvenc"), ("H.264(CPU)", "libx264")]
RATIOS = [("自由", 0), ("16:9", 16/9), ("4:3", 4/3), ("1:1", 1), ("3:2", 3/2), ("21:9", 21/9), ("16:10", 16/10), ("9:16竖屏", 9/16), ("3:4竖屏", 3/4)]
SEGMENT_OPTIONS = [("全时长", 0), ("≈3分钟", 180), ("≈5分钟", 300), ("≈10分钟", 600)]
STANDARD_SIZES = [("8K UHD",7680,4320),("4K UHD",3840,2160),("1440p",2560,1440),("1080p横",1920,1080),("1080p竖",1080,1920),("720p横",1280,720),("720p竖",720,1280),("方形1080",1080,1080),("方形720",720,720),("方形480",480,480),("VGA",640,480)]
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".ts", ".m4v")


def _fmt(s):
    s = max(0, int(s)); return f"{s//60:02d}:{s%60:02d}"


# ── FramePrefetcher ────────────────────────────────────────────────────────────
class FramePrefetcher:
    _STOP = object()
    def __init__(self, cap, buf=8):
        self._cap = cap
        self._q = queue.Queue(maxsize=buf)
        self._alive = threading.Event(); self._alive.set()
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
    def _run(self):
        while self._alive.is_set():
            ret, frame = self._cap.read()
            if not self._alive.is_set(): return
            if not ret:
                try: self._q.put(self._STOP, timeout=1)
                except queue.Full: pass
                return
            while self._alive.is_set():
                try: self._q.put(frame, timeout=0.05); break
                except queue.Full: time.sleep(0.01)
    def get(self):
        try: return self._q.get_nowait()
        except queue.Empty: return None
    def stop(self):
        self._alive.clear()
        while True:
            try: self._q.get_nowait()
            except queue.Empty: break
        self._t.join(timeout=1.0)


# ── RangeSlider ────────────────────────────────────────────────────────────────
class RangeSlider(QWidget):
    range_changed = Signal(float, float)
    position_changed = Signal(float)
    position_clicked = Signal(float)
    interaction_ended = Signal()
    HANDLE_W, TRACK_H, MARGIN = 8, 22, 10
    _DRAG_THRESHOLD = 5  # 像素，区分点击与拖动的阈值
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(36); self.setMouseTracking(True)
        self._range_start = self._range_end = 0.0; self._position = 0.0
        self._drag_mode = None; self._drag_off = 0.0; self._pending_range = None
    def set_range(self, s, e):
        s, e = max(0,min(s,1)), max(0,min(e,1))
        self._range_start, self._range_end = min(s,e), max(s,e); self.update()
    def set_position(self, p):
        self._position = max(0, min(p, 1)); self.update()
    def _tr(self):
        m = self.MARGIN; return QRectF(m, 6, self.width()-m*2, self.TRACK_H)
    def _x2r(self, x):
        r = self._tr(); return 0 if r.width()<=0 else max(0, min((x-r.left())/r.width(), 1))
    def _r2x(self, r):
        rect = self._tr(); return rect.left() + r*rect.width()
    def _hr(self, r):
        cx = self._r2x(r); rect = self._tr(); hw = self.HANDLE_W
        return QRectF(cx-hw/2, rect.top()-2, hw, rect.height()+4)
    def _pr(self):
        cx = self._r2x(self._position); rect = self._tr()
        return QRectF(cx-3, rect.top()-8, 6, rect.height()+12)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._tr()
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor("#3a3a3a")); p.drawRoundedRect(rect, 3, 3)
        if self._range_end > self._range_start + 0.001:
            sx, ex = self._r2x(self._range_start), self._r2x(self._range_end)
            sel = QRectF(sx, rect.top(), ex-sx, rect.height())
            p.setBrush(QColor("#00e676")); p.drawRoundedRect(sel, 3, 3)
        for name, r in (("left",self._range_start),("right",self._range_end)):
            hr = self._hr(r)
            p.setBrush(QColor("#00e676")); p.setPen(QPen(QColor("#00b060"),1)); p.drawRoundedRect(hr, 2, 2)
            cx = hr.center().x()
            tri = QPolygonF([QPointF(cx-4,hr.top()-1), QPointF(cx+4,hr.top()-1), QPointF(cx,hr.top()-6)])
            p.setBrush(QColor("#00e676")); p.setPen(Qt.PenStyle.NoPen); p.drawPolygon(tri)
        px = self._r2x(self._position)
        p.setPen(QPen(QColor("#ff5252"), 2)); p.drawLine(QPointF(px, rect.top()-4), QPointF(px, rect.bottom()+2))
        p.setBrush(QColor("#ff5252")); p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(QPointF(px, rect.top()-6), 4, 4)
        p.end()
    def _pick(self, pos):
        for name, r in (("left",self._range_start),("right",self._range_end)):
            if self._hr(r).adjusted(-2,0,2,0).contains(pos): return name
        return None
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            pos = e.position() if hasattr(e,'position') else e.localPos()
            h = self._pick(pos)
            if h:
                self._drag_mode = h; self._drag_off = self._x2r(pos.x()) - (self._range_start if h=="left" else self._range_end); return
            if self._pr().contains(pos): self._drag_mode = "playhead"; return
            r = self._x2r(pos.x())
            if self._range_end - self._range_start < 0.999 and self._range_start < r < self._range_end:
                self._pending_range = r; self._drag_off = r - self._range_start; return
            self._position = r; self.position_clicked.emit(self._position); self.update()
        super().mousePressEvent(e)
    def mouseMoveEvent(self, e):
        pos = e.position() if hasattr(e,'position') else e.localPos(); r = self._x2r(pos.x())
        if not (e.buttons() & Qt.MouseButton.LeftButton):
            if self._pick(pos) or self._pr().contains(pos): self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._tr().contains(pos): self.setCursor(Qt.CursorShape.PointingHandCursor)
            else: self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._drag_mode == "left": self._range_start = max(0, min(r, self._range_end-0.01)); self.range_changed.emit(self._range_start, self._range_end)
        elif self._drag_mode == "right": self._range_end = max(self._range_start+0.01, min(r,1)); self.range_changed.emit(self._range_start, self._range_end)
        elif self._drag_mode == "range":
            d = r - self._drag_off; span = self._range_end - self._range_start
            if d >= 0 and d+span <= 1: self._range_start = d; self._range_end = d+span; self.range_changed.emit(self._range_start, self._range_end)
        elif self._drag_mode == "playhead": self._position = max(0, min(r,1)); self.position_changed.emit(self._position)
        elif self._pending_range is not None:
            if abs(r - self._pending_range) * self._tr().width() > self._DRAG_THRESHOLD:
                self._drag_mode = "range"; self._pending_range = None
                d = r - self._drag_off; span = self._range_end - self._range_start
                if d >= 0 and d+span <= 1: self._range_start = d; self._range_end = d+span; self.range_changed.emit(self._range_start, self._range_end)
        self.update(); super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._pending_range is not None:
                r = self._pending_range; self._pending_range = None; self._drag_off = 0.0
                self._position = r; self.position_clicked.emit(self._position); self.update(); return
            if self._drag_mode:
                self._drag_mode = None; self._drag_off = 0.0; self.interaction_ended.emit()
        super().mouseReleaseEvent(e)


# ── TimeCodeSpinBox ────────────────────────────────────────────────────────────
class TimeCodeSpinBox(QSpinBox):
    userEdited = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 86399); self.setSingleStep(1); self.setDisplayIntegerBase(10)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QSpinBox{font-family:Consolas,monospace;font-size:14px;color:#00e676;background:#2a2a2a;border:1px solid #444;border-radius:3px;padding:2px 2px;min-width:0;max-width:76px}QSpinBox::up-button,QSpinBox::down-button{width:0;height:0;subcontrol-origin:margin}")
        self._prog = False; self.valueChanged.connect(self._on_v)
    def _on_v(self, v):
        if not self._prog: self.userEdited.emit(v)
    def stepBy(self, steps):
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: steps *= 60
        super().stepBy(steps)
    def textFromValue(self, v): v = max(0,int(v)); return f"{v//3600:02d}:{(v%3600)//60:02d}:{v%60:02d}"
    def valueFromText(self, t):
        parts = t.strip().split(":")
        try: nums = [int(p) if p else 0 for p in parts]
        except ValueError: return self.value()
        if len(nums)==3: return nums[0]*3600+nums[1]*60+nums[2]
        if len(nums)==2: return nums[0]*60+nums[1]
        if len(nums)==1: return nums[0]
        return self.value()
    _F = re.compile(r"^\d{2}:\d{2}:\d{2}$"); _P = re.compile(r"^\d{0,2}:?\d{0,2}:?\d{0,2}$")
    def validate(self, text, pos):
        if self._F.match(text): return QValidator.State.Acceptable, text, pos
        if self._P.match(text): return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos
    def setTimeCode(self, sec):
        self._prog = True
        try: self.setValue(max(0, min(int(sec), self.maximum())))
        finally: self._prog = False
    def timeCode(self): return self.value()


# ── VideoView ──────────────────────────────────────────────────────────────────
class VideoView(QGraphicsView):
    crop_changed = Signal(int,int,int,int)
    HANDLE_HALF = 10
    _P = {"tl":(0,0),"tc":(.5,0),"tr":(1,0),"ml":(0,.5),"mr":(1,.5),"bl":(0,1),"bc":(.5,1),"br":(1,1)}
    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.RenderHint.Antialiasing|QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag); self.setAcceptDrops(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color:#111;border:none"); self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self._pix = self._rect = self._text = None; self._handles = []
        self._dmode = "none"; self._dh = None; self._ds = None; self._da = None
        self._iw = self._ih = 0; self._ratio = 0.0; self._crop = None; self._interact = True
    def _hrects(self, r):
        hh = self.HANDLE_HALF; return {n: QRectF(r.x()+r.width()*rx-hh, r.y()+r.height()*ry-hh, hh*2, hh*2) for n,(rx,ry) in self._P.items()}
    def _draw_h(self, r):
        for i in self._handles: self._scene.removeItem(i)
        self._handles = []
        f = QColor("#00e676"); f.setAlpha(180); pen = QPen(QColor("#00e676"),1); pen.setCosmetic(True)
        for hr in self._hrects(r).values():
            i = self._scene.addRect(hr, pen, QBrush(f)); i.setZValue(12); self._handles.append(i)
    def _hc(self, n): return {"tl":Qt.CursorShape.SizeFDiagCursor,"br":Qt.CursorShape.SizeFDiagCursor,"tr":Qt.CursorShape.SizeBDiagCursor,"bl":Qt.CursorShape.SizeBDiagCursor,"tc":Qt.CursorShape.SizeVerCursor,"bc":Qt.CursorShape.SizeVerCursor,"ml":Qt.CursorShape.SizeHorCursor,"mr":Qt.CursorShape.SizeHorCursor}.get(n, Qt.CursorShape.ArrowCursor)
    def _ha(self, sp):
        if not self._crop: return None
        for n, hr in self._hrects(self._crop).items():
            if hr.contains(sp): return n
        return None
    def _snap(self, x1, y1, x2, y2):
        if self._ratio > 0:
            rw, rh = abs(x2-x1), abs(y2-y1)
            if rw/max(1,rh) > self._ratio: x2 = x1 + int(rh*self._ratio)*(1 if x2>=x1 else -1)
            else: y2 = y1 + int(rw/self._ratio)*(1 if y2>=y1 else -1)
        return int(x1),int(y1),int(max(0,min(x2,self._iw))),int(max(0,min(y2,self._ih)))
    def init_frame(self, qi):
        self._scene.clear(); self._pix = self._scene.addPixmap(QPixmap.fromImage(qi))
        self._rect = self._text = None; self._handles = []
        self._dmode = "none"; self._dh = self._ds = self._da = None; self._crop = None
        self._iw, self._ih = qi.width(), qi.height(); self._scene.setSceneRect(0,0,self._iw,self._ih)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def reset_view(self):
        self._scene.clear()
        self._pix = self._rect = self._text = None
        self._handles = []
        self._dmode = "none"; self._dh = self._ds = self._da = None
        self._crop = None; self._iw = self._ih = 0
    def update_frame(self, qi):
        if self._pix is None: self.init_frame(qi); return
        self._pix.setPixmap(QPixmap.fromImage(qi))
    def set_ratio(self, r):
        self._ratio = r; cr = self._crop
        if cr is None: return
        x,y,w,h = int(cr.x()),int(cr.y()),int(cr.width()),int(cr.height())
        if r > 0:
            if w/max(1,h) > r: w = int(h*r)
            else: h = int(w/r)
        self._ap(x,y,w,h)
    def _ap(self, x, y, w, h):
        for i in (self._rect, self._text):
            if i: self._scene.removeItem(i)
        for i in self._handles: self._scene.removeItem(i)
        self._rect = self._text = None; self._handles = []; self._crop = None
        if w < 2 or h < 2: self.crop_changed.emit(0,0,0,0); return
        x = max(0,min(x,self._iw-2)); y = max(0,min(y,self._ih-2))
        w = max(2,min(w,self._iw-x)); h = max(2,min(h,self._ih-y))
        pen = QPen(QColor("#00e676"),2); pen.setCosmetic(True)
        self._rect = self._scene.addRect(x,y,w,h,pen); self._rect.setZValue(10)
        f = QFont("Consolas",11); self._text = self._scene.addText(f" {w}x{h}  ({x},{y}) ",f)
        self._text.setDefaultTextColor(QColor("#00e676"))
        lh = self._text.boundingRect().height()
        self._text.setPos(x+2, y-lh-2 if y>lh+4 else y+4); self._text.setZValue(13)
        self._crop = QRectF(x,y,w,h); self._draw_h(self._crop); self.crop_changed.emit(x,y,w,h)
    def clear_crop(self):
        if not self._interact: return
        self._dmode = "none"; self._dh = self._ds = self._da = None; self._ap(0,0,0,0)
    def set_crop_exact(self, x, y, w, h):
        if self._iw <= 0: return
        self._dmode = "none"; self._dh = self._ds = self._da = None
        if self._ratio > 0:
            if w/max(1,h) > self._ratio: w = int(h*self._ratio)
            else: h = int(w/self._ratio)
        self._ap(x,y,w,h)
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._iw > 0: self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def mousePressEvent(self, e):
        if not self._interact: return
        if e.button() == Qt.MouseButton.LeftButton and self._iw > 0:
            sp = self.mapToScene(e.pos())
            sx, sy = max(0,min(int(sp.x()),self._iw)), max(0,min(int(sp.y()),self._ih))
            h = self._ha(sp)
            if h and self._crop:
                self._dmode = "resize"; self._dh = h; r = self._crop
                anchors = {"tl":(r.right(),r.bottom()),"tc":(r.center().x(),r.bottom()),"tr":(r.x(),r.bottom()),"ml":(r.right(),r.center().y()),"mr":(r.x(),r.center().y()),"bl":(r.right(),r.y()),"bc":(r.center().x(),r.y()),"br":(r.x(),r.y())}
                self._da = anchors[h]; e.accept(); return
            if self._crop and self._crop.contains(sp):
                self._dmode = "move"; self._ds = (sx,sy); e.accept(); return
            self._dmode = "create"; self._ds = (sx,sy); e.accept(); return
        super().mousePressEvent(e)
    def mouseMoveEvent(self, e):
        sp = self.mapToScene(e.pos())
        sx, sy = max(0,min(int(sp.x()),self._iw)), max(0,min(int(sp.y()),self._ih))
        hb = bool(e.buttons() & Qt.MouseButton.LeftButton)
        if not hb and self._iw > 0:
            if not self._interact: self.setCursor(Qt.CursorShape.ArrowCursor); super().mouseMoveEvent(e); return
            h = self._ha(sp)
            if h: self.setCursor(self._hc(h))
            elif self._crop and self._crop.contains(sp): self.setCursor(Qt.CursorShape.SizeAllCursor)
            else: self.setCursor(Qt.CursorShape.CrossCursor)
        if not hb: super().mouseMoveEvent(e); return
        if self._dmode == "resize" and self._dh and self._da:
            ax, ay = self._da; r = self._crop
            if self._dh == "tc": self._ap(int(r.x()), min(sy,int(ay)), int(r.width()), int(ay)-min(sy,int(ay)))
            elif self._dh == "bc": self._ap(int(r.x()), int(r.y()), int(r.width()), max(2,abs(sy-int(ay))))
            elif self._dh == "ml": self._ap(min(sx,int(ax)), int(r.y()), int(ax)-min(sx,int(ax)), int(r.height()))
            elif self._dh == "mr": self._ap(int(ax), int(r.y()), max(2,abs(sx-int(ax))), int(r.height()))
            else:
                if self._ratio > 0:
                    mx, my = sx, sy
                    if abs(mx-int(ax))/max(1,abs(my-int(ay))) > self._ratio: mx = int(ax) + int((my-int(ay))*self._ratio)*(1 if mx>=int(ax) else -1)
                    else: my = int(ay) + int((mx-int(ax))/self._ratio)*(1 if my>=int(ay) else -1)
                    nx, ny = min(mx,int(ax)), min(my,int(ay)); nw, nh = abs(mx-int(ax)), abs(my-int(ay))
                else: nx, ny = min(sx,int(ax)), min(sy,int(ay)); nw, nh = abs(sx-int(ax)), abs(sy-int(ay))
                self._ap(nx,ny,nw,nh)
            e.accept(); return
        if self._dmode == "move" and self._ds:
            r = self._crop; dx = sx-self._ds[0]; dy = sy-self._ds[1]
            if r is None: return
            nx = max(0, min(int(r.x()+dx), self._iw-int(r.width())))
            ny = max(0, min(int(r.y()+dy), self._ih-int(r.height())))
            self._ap(nx, ny, int(r.width()), int(r.height()))
            self._ds = (sx,sy); e.accept(); return
        if self._dmode == "create" and self._ds:
            x1,y1 = self._ds; _,_,x2,y2 = self._snap(x1,y1,sx,sy)
            self._ap(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)); e.accept(); return
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._dmode != "none":
            self._dmode = "none"; self._dh = self._ds = self._da = None
            self.setCursor(Qt.CursorShape.ArrowCursor); e.accept(); return
        super().mouseReleaseEvent(e)


# ── MainWindow ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────
# 批处理面板
# ──────────────────────────────────────────────
class BatchPanel(QWidget):
    video_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumWidth(200); self.setMaximumWidth(500)
        self._videos = []; self._ref_ratio = None
        self._thumb_size = QSize(120, 68); self._grid_size = QSize(125, 105)
        self.setStyleSheet("BatchPanel{background-color:#1e1e1e;border-right:1px solid #333}")
        ly = QVBoxLayout(self); ly.setContentsMargins(4,4,4,4); ly.setSpacing(4)
        h = QHBoxLayout()
        self.lbl_count = QLabel("0 个视频"); h.addWidget(self.lbl_count); h.addStretch()
        self.btn_toggle = QPushButton("📋列表"); self.btn_toggle.setCheckable(True); self.btn_toggle.setFixedWidth(80)
        self.btn_toggle.clicked.connect(self._toggle_view); h.addWidget(self.btn_toggle)
        self.btn_close = QPushButton("✕关闭"); self.btn_close.setFixedWidth(60)
        self.btn_close.clicked.connect(self.hide); h.addWidget(self.btn_close)
        ly.addLayout(h)
        self._list = QListWidget(); self._list.setIconSize(self._thumb_size)
        self._list.setViewMode(QListWidget.IconMode); self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setGridSize(self._grid_size); self._list.setSpacing(2)
        self._list.setStyleSheet("QListWidget{background-color:#222;border:1px solid #333}")
        self._list.itemClicked.connect(self._on_item_clicked); ly.addWidget(self._list, 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._list.viewMode() == QListWidget.IconMode:
            self._update_grid()
        else:
            self._refresh()  # 列表模式重新截断文件名

    def _update_grid(self):
        aw = self._list.viewport().width() - 4
        cols = 2  # 始终两列
        cw = (aw - (cols-1)*2) // cols
        tw = max(80, cw - 8); th = int(tw * 9 / 16)
        self._thumb_size = QSize(tw, th); self._grid_size = QSize(cw, th + 36)
        self._list.setIconSize(self._thumb_size); self._list.setGridSize(self._grid_size)

    def _toggle_view(self):
        if self._list.viewMode() == QListWidget.IconMode:
            self._list.setViewMode(QListWidget.ListMode); self._list.setGridSize(QSize())
            self._list.setIconSize(QSize(32,18)); self.btn_toggle.setText("🖼缩略图")
        else:
            self._list.setViewMode(QListWidget.IconMode); self._update_grid(); self.btn_toggle.setText("📋列表")
        self._refresh()

    def _on_item_clicked(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p: self.video_selected.emit(p)

    def load_folder(self, folder_path, ref_ratio=None):
        self._ref_ratio = ref_ratio; self._videos = []; self._list.clear()
        folder = Path(folder_path)
        if not folder.is_dir(): return
        files = []
        for ext in VIDEO_EXT: files.extend(folder.glob(f"*{ext}"))
        for vf in sorted(files): self._videos.append(self._probe(vf))
        self._refresh(); self.lbl_count.setText(f"{len(self._videos)} 个视频")

    def _probe(self, vf):
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened(): return {"path":str(vf),"ratio":0,"status":"读取失败","thumb":None,"size":"?"}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w==0 or h==0: cap.release(); return {"path":str(vf),"ratio":0,"status":"读取失败","thumb":None,"size":"?"}
        ratio = w/h
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read(); thumb = None
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); fh, fw = frame.shape[:2]
            sc = min(240/fw, 240/fh); nw, nh = int(fw*sc), int(fh*sc)
            frame = cv2.resize(frame, (nw, nh))
            qimg = QImage(frame.data, nw, nh, nw*3, QImage.Format.Format_RGB888)
            thumb = QPixmap.fromImage(qimg.copy())
        cap.release()
        status = ""
        if self._ref_ratio is not None and abs(ratio - self._ref_ratio) / max(self._ref_ratio, 0.001) > 0.03: status = "尺寸不匹配"
        return {"path":str(vf),"ratio":ratio,"status":status,"thumb":thumb,"size":f"{w}x{h}"}

    def _list_max_name_len(self):
        """列表模式：根据面板宽度计算文件名最大字符数"""
        aw = self._list.viewport().width() - 36  # 图标+边距
        # 尺寸部分约10字符，状态部分约12字符
        fixed = 22 if any(v["status"] for v in self._videos) else 10
        return max(8, aw // 8 - fixed)

    def _refresh(self):
        self._list.clear()
        is_icon = self._list.viewMode() == QListWidget.IconMode
        list_max = self._list_max_name_len() if not is_icon else 0
        for v in self._videos:
            name = Path(v["path"]).name
            if is_icon:
                # 缩略图模式：始终2行（文件名+尺寸），状态通过颜色区分
                if len(name) > 18: name = name[:15] + "…"
                display = f"{name}\n{v['size']}"
            else:
                # 列表模式：根据面板宽度截断文件名，保证状态始终可见
                if len(name) > list_max: name = name[:list_max-3] + "…"
                display = f"{name}  {v['size']}"
                if v["status"]: display += f"  [{v['status']}]"
            item = QListWidgetItem(); item.setText(display); item.setData(Qt.ItemDataRole.UserRole, v["path"])
            if v["thumb"]: item.setIcon(QIcon(v["thumb"]))
            if v["status"] == "尺寸不匹配": item.setForeground(QColor("#ff6b6b"))
            elif v["status"] == "已处理": item.setForeground(QColor("#00e676"))
            self._list.addItem(item)

    def mark_done(self, path):
        for v in self._videos:
            if v["path"] == path: v["status"] = "已处理"; break
        self._refresh()

    def set_ref_ratio(self, ratio):
        self._ref_ratio = ratio
        for v in self._videos:
            if v["status"] == "已处理": continue
            v["status"] = "尺寸不匹配" if abs(v["ratio"]-ratio)/max(ratio,0.001)>0.03 else ""
        self._refresh()


class MainWindow(QMainWindow):
    sig_progress = Signal(int); sig_done = Signal(str); sig_done_msg = Signal(str)
    sig_error = Signal(str); sig_status = Signal(str); sig_finish_exp = Signal()
    sig_mark_batch_done = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFmpeg GPU 裁剪工具"); self.resize(1100, 760); self.setAcceptDrops(True)
        self.cap = None; self.video_path = None; self.total_frames = 0; self.fps = 30.0
        self.cur_frame = 0; self.img_w = self.img_h = 0
        self.crop_x = self.crop_y = self.crop_w = self.crop_h = 0
        self._playing = False; self._play_speed = 1.0; self._play_acc = 0.0
        self._prefetcher = None; self._ffmpeg_proc = None
        self._exporting = False; self._export_cur_sec = 0.0; self._cancel_req = False
        self._multi_segment = False; self._multi_msg = ""
        self._export_thread = None; self._range_s = self._range_e = 0.0
        self._lock = threading.Lock()
        self._pend_seek = -1; self._pend_move = True
        self._seek_timer = QTimer(self); self._seek_timer.setSingleShot(True); self._seek_timer.setInterval(80)
        self._seek_timer.timeout.connect(self._flush_seek)
        self._play_timer = QTimer(self); self._play_timer.timeout.connect(self._play_tick)
        self.sig_progress.connect(lambda v: self.progress.setValue(v))
        self.sig_done.connect(lambda p: self._on_export_done(p))
        self.sig_done_msg.connect(self._on_export_done_msg)
        self.sig_error.connect(lambda m: QMessageBox.critical(self, "错误", m))
        self.sig_status.connect(self._on_status)
        self.sig_finish_exp.connect(self._finish_exp)
        self._setup_ui(); self._setup_keys()
        self.sig_mark_batch_done.connect(self.batch_panel.mark_done)
        if not FFMPEG: QMessageBox.critical(self, "错误", "未找到 ffmpeg")

    def _setup_ui(self):
        cw = QWidget(); self.setCentralWidget(cw); v = QVBoxLayout(cw); v.setContentsMargins(8,8,8,8); v.setSpacing(6)
        self.view = VideoView(); self.view.crop_changed.connect(self._on_crop)
        tb = QHBoxLayout(); tb.setSpacing(6)
        self.btn_open = QPushButton("📂 打开"); self.btn_open.clicked.connect(self._open); tb.addWidget(self.btn_open)
        self.btn_batch = QPushButton("📁 批处理"); self.btn_batch.clicked.connect(self._on_batch); tb.addWidget(self.btn_batch)
        tb.addWidget(QLabel("约束:"))
        self.combo_ratio = QComboBox()
        for n,_ in RATIOS: self.combo_ratio.addItem(n)
        self.combo_ratio.currentIndexChanged.connect(lambda i: self.view.set_ratio(RATIOS[i][1]))
        tb.addWidget(self.combo_ratio)
        tb.addWidget(QLabel("规格:"))
        self.combo_preset = QComboBox(); self.combo_preset.setMinimumWidth(100)
        self.combo_preset.currentIndexChanged.connect(self._on_preset); tb.addWidget(self.combo_preset)
        self.btn_clear = QPushButton("✕清除"); self.btn_clear.clicked.connect(self.view.clear_crop); tb.addWidget(self.btn_clear)
        tb.addWidget(QLabel("编码:"))
        self.combo_enc = QComboBox()
        for n,_ in ENC_OPTIONS: self.combo_enc.addItem(n)
        tb.addWidget(self.combo_enc)
        tb.addWidget(QLabel("分段:"))
        self.combo_seg = QComboBox()
        for n,_ in SEGMENT_OPTIONS: self.combo_seg.addItem(n)
        self.combo_seg.setFixedWidth(80); tb.addWidget(self.combo_seg)
        self.btn_crop = QPushButton("✂导出裁剪"); self.btn_crop.setEnabled(bool(FFMPEG))
        self.btn_crop.clicked.connect(self._do_crop); tb.addWidget(self.btn_crop)
        self.btn_batch_apply = QPushButton("批量应用并处理"); self.btn_batch_apply.setEnabled(bool(FFMPEG))
        self.btn_batch_apply.clicked.connect(self._on_batch_apply); self.btn_batch_apply.hide(); tb.addWidget(self.btn_batch_apply)
        self.lbl_info = QLabel("拖入视频文件"); self.lbl_info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(self.lbl_info, 1); v.addLayout(tb)

        # 分割布局：左侧批处理面板 + 右侧主内容
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.batch_panel = BatchPanel()
        self.batch_panel.video_selected.connect(self._on_batch_video_select)
        self.batch_panel.btn_close.clicked.connect(self._hide_batch)
        self.batch_panel.hide()
        self.splitter.addWidget(self.batch_panel)

        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0); rv.setSpacing(6)
        rv.addWidget(self.view, 1)
        self.lbl_coords = QLabel("拖拽画面选择裁剪区域"); self.lbl_coords.setStyleSheet("color:#aaa;font-size:12px"); rv.addWidget(self.lbl_coords)
        c = QHBoxLayout(); c.setSpacing(8)
        self.lbl_time = QLabel("00:00/00:00"); self.lbl_time.setFixedWidth(120)
        self.lbl_time.setStyleSheet("font-family:Consolas,monospace;font-size:14px;color:#aaa"); c.addWidget(self.lbl_time)
        self.btn_play = QPushButton(); self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.setStyleSheet("QPushButton{background-color:#3a3a3a;border:none;border-radius:6px;padding:0;min-width:40px;min-height:36px}QPushButton:hover{background-color:#4a4a4a}QPushButton:pressed{background-color:#2a2a2a}QPushButton:disabled{background-color:#222}")
        self.btn_play.setToolTip("播放/暂停 (Space)"); self.btn_play.clicked.connect(self._toggle_play); c.addWidget(self.btn_play)
        _ls = QLabel("速度:"); _ls.setStyleSheet("font-size:13px"); c.addWidget(_ls)
        self._SPDS = [("0.25x",.25),("0.5x",.5),("1x",1),("1.5x",1.5),("2x",2),("4x",4)]
        self.combo_spd = QComboBox()
        for n,_ in self._SPDS: self.combo_spd.addItem(n)
        self.combo_spd.setCurrentIndex(2); self.combo_spd.setFixedWidth(60)
        self.combo_spd.currentIndexChanged.connect(self._on_spd); c.addWidget(self.combo_spd)
        c.addSpacing(16)
        _li = QLabel("范围入点"); _li.setStyleSheet("font-size:13px"); c.addWidget(_li)
        self.edit_in = TimeCodeSpinBox(); self.edit_in.userEdited.connect(lambda s: self._apply_in(s)); c.addWidget(self.edit_in); c.addSpacing(10)
        _lo = QLabel("范围出点"); _lo.setStyleSheet("font-size:13px"); c.addWidget(_lo)
        self.edit_out = TimeCodeSpinBox(); self.edit_out.userEdited.connect(lambda s: self._apply_out(s)); c.addWidget(self.edit_out); c.addSpacing(10)
        _ld = QLabel("范围时长"); _ld.setStyleSheet("font-size:13px"); c.addWidget(_ld)
        self.lbl_sel = QLabel("--:--"); self.lbl_sel.setStyleSheet("font-family:Consolas,monospace;font-size:14px;color:#00e676;padding:0 4px"); c.addWidget(self.lbl_sel)
        c.addStretch(); rv.addLayout(c)
        self.slider = RangeSlider()
        self.slider.position_clicked.connect(lambda r: self._show_imm(int(r*self.total_frames)))
        self.slider.position_changed.connect(lambda r: self._show(int(r*self.total_frames)))
        self.slider.range_changed.connect(self._on_slider)
        self.slider.interaction_ended.connect(self._flush_seek)
        rv.addWidget(self.slider)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        rv.addWidget(self.progress)
        self.splitter.addWidget(right)
        v.addWidget(self.splitter, 1)
        self.status = QStatusBar()
        hint = f"检测到 {NVENC_ENC}" if NVENC_ENC else "未检测到 NVENC"
        self.status.showMessage(f"就绪 | {hint} | Space 播放 ←→逐帧 Esc清除框")
        self.setStatusBar(self.status)

    def _setup_keys(self):
        QShortcut(QKeySequence("Space"), self, self._toggle_play)
        QShortcut(QKeySequence("Left"), self, lambda: self._step(-1))
        QShortcut(QKeySequence("Right"), self, lambda: self._step(1))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self._step(-10))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self._step(10))
        QShortcut(QKeySequence("Home"), self, lambda: self._jump(0))
        QShortcut(QKeySequence("End"), self, lambda: self._jump(1))
        QShortcut(QKeySequence("Escape"), self, self.view.clear_crop)
        QShortcut(QKeySequence("["), self, lambda: self._adj_spd(-1))
        QShortcut(QKeySequence("]"), self, lambda: self._adj_spd(1))

    def _jump(self, which):
        if not self.cap: return
        self._stop()
        self._show_imm(int((self._range_s if which==0 else self._range_e) * self.total_frames) - (0 if which==0 else 1))

    # ── 拖放 / 打开 ──
    def dragEnterEvent(self, e):
        if self._exporting: return
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile() and u.toLocalFile().lower().endswith(VIDEO_EXT): e.acceptProposedAction(); return
    def dropEvent(self, e):
        if self._exporting: return
        for u in e.mimeData().urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if p.lower().endswith(VIDEO_EXT):
                    if self.batch_panel.isVisible():
                        self._hide_batch()
                    self._load(p); return
    def _open(self):
        p,_ = QFileDialog.getOpenFileName(self, "选择视频", "", f"视频(*{' *'.join(VIDEO_EXT)});;所有文件(*)")
        if p:
            if self.batch_panel.isVisible():
                self._hide_batch()
            self._load(p)

    def _on_batch(self):
        d = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if not d: return
        ref = self.img_w/self.img_h if self.img_w > 0 else None
        self._stop()
        self.batch_panel.load_folder(d, ref)
        if not self.batch_panel._videos:
            QMessageBox.information(self, "提示", "文件夹中没有找到视频文件"); return
        self.batch_panel.show()
        self.splitter.setSizes([280, self.splitter.width()-280])
        self.btn_crop.setText("✂导出预览视频")
        self.btn_batch_apply.show()
        # 自动加载列表中第一个可播放的视频
        for v in self.batch_panel._videos:
            if v["status"] != "读取失败":
                self._load(v["path"])
                if self.img_w > 0:
                    self.batch_panel.set_ref_ratio(self.img_w / self.img_h)
                break
        else:
            # 所有视频都读取失败
            self.view.reset_view()

    def _on_batch_video_select(self, path):
        if self.video_path == path: return
        self._load(path)
        if self.img_w > 0:
            self.batch_panel.set_ref_ratio(self.img_w / self.img_h)

    def _hide_batch(self):
        self._stop()
        self.batch_panel.hide()
        self.btn_crop.setText("✂导出裁剪")
        self.btn_batch_apply.hide()
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_batch_apply(self):
        if self._exporting or not self.video_path: return
        if self.crop_w <= 0 or self.crop_h <= 0:
            QMessageBox.warning(self, "提示", "请先在预览画面中选择裁剪区域"); return
        eligible = [v for v in self.batch_panel._videos if v["status"] != "尺寸不匹配" and v["status"] != "已处理" and v["status"] != "读取失败"]
        if not eligible: QMessageBox.information(self, "提示", "没有可处理的视频"); return
        output_dir = str(Path(self.video_path).parent / "crop_output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        enc = self._get_enc()
        src_w, src_h = self.img_w, self.img_h
        cx, cy, cw, ch = self.crop_x, self.crop_y, self.crop_w, self.crop_h
        self._set_lock(True); self._exporting = True; self._cancel_req = False; self._export_cur_sec = 0
        self.btn_crop.setText("✕取消"); self.btn_crop.setEnabled(True)
        try: self.btn_crop.clicked.disconnect()
        except (TypeError, RuntimeError): pass
        self.btn_crop.clicked.connect(self._cancel_exp)
        self.progress.setVisible(True); self.progress.setValue(0)
        self.sig_status.emit("正在批量处理…")
        paths = [v["path"] for v in eligible]
        self._export_thread = threading.Thread(target=self._run_batch, args=(paths, output_dir, enc, src_w, src_h, cx, cy, cw, ch), daemon=True)
        self._export_thread.start()

    def _run_batch(self, paths, output_dir, enc, src_w, src_h, cx, cy, cw, ch):
        total = len(paths)
        for i, path in enumerate(paths):
            with self._lock:
                if self._cancel_req: break
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): continue
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw == 0 or vh == 0: continue
            sx = vw / src_w; sy = vh / src_h
            bcx = int(cx * sx); bcy = int(cy * sy); bcw = int(cw * sx); bch = int(ch * sy)
            name = Path(path).stem
            out = str(Path(output_dir) / f"{name}_crop.mp4")
            bp = int(i/total*100); np2 = int((i+1)/total*100)
            ok = self._exec_ff(path, out, enc, crop=(bcx,bcy,bcw,bch), cb=lambda p,b=bp,np=np2: self.sig_progress.emit(b+int(p/100*(np-b))))
            if ok: self.sig_mark_batch_done.emit(path)
            self._multi_msg = f"批量 {i+1}/{total}  {name}"; self.sig_status.emit(self._multi_msg)
        self.sig_progress.emit(100)
        self.sig_done_msg.emit(f"批量完成：{total} 个视频 → {Path(output_dir).name}")
        self.sig_finish_exp.emit()

    def _get_enc(self):
        idx = self.combo_enc.currentIndex(); chosen = ENC_OPTIONS[idx][1]
        if chosen is None or (chosen.endswith("_nvenc") and chosen != NVENC_ENC): return NVENC_ENC or "libx264"
        return chosen

    # ── 加载 ──
    def _load(self, path):
        self._stop(); self._del_pf(); self.view.setEnabled(True)
        # 清理可能残留的导出锁定状态
        if self._exporting:
            self._restore_btn()
        if self.cap: self.cap.release()
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            QMessageBox.warning(self, "错误", f"无法读取：{path}"); cap.release(); return
        self.cap = cap; self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(1, cap.get(cv2.CAP_PROP_FPS)); self.img_w, self.img_h = w, h
        self.cur_frame = 0; self.video_path = path
        self.setWindowTitle(f"裁剪 — {Path(path).name}  {w}×{h}  {self.total_frames}帧  {self.fps:.2f}fps")
        self.lbl_info.setText("就绪")
        self.slider.set_range(0, 1); self.slider.set_position(0); self.slider.update()
        self._range_s, self._range_e = 0, 1
        self.view.clear_crop(); self.combo_ratio.setCurrentIndex(0); self._ref_presets()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        if not self.batch_panel.isVisible():
            nw = min(max(w+40,640),1400); nh = min(max(h+200,480),900); self.resize(nw,nh)
        self.view.reset_view(); self._show_imm(0)

    # ── 帧显示 ──
    def _show(self, idx):
        """跳转（带 80ms 节流）"""
        if not self.cap: return
        if self._playing: self._stop()
        idx = max(0, min(idx, self.total_frames-1))
        self._pend_seek = idx; self._pend_move = True; self._seek_timer.start()

    def _show_imm(self, idx):
        """立即跳转（快捷键单步）"""
        if not self.cap: return
        self._seek_timer.stop(); self._pend_seek = -1
        if self._playing: self._stop()
        self._del_pf()
        idx = max(0, min(idx, self.total_frames-1))
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx): return
        r, f = self.cap.read()
        if not r: return
        self.cur_frame = idx; self._disp(f)

    def _show_nopf(self, idx):
        """跳转但不更新播放头"""
        if not self.cap: return
        idx = max(0, min(idx, self.total_frames-1))
        self._pend_seek = idx; self._pend_move = False; self._seek_timer.start()

    def _flush_seek(self):
        idx = self._pend_seek
        if idx < 0 or not self.cap: return
        self._pend_seek = -1
        if self._playing: self._stop()
        self._del_pf()
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx): return
        r, f = self.cap.read()
        if not r: return
        self.cur_frame = idx
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
        self.view.update_frame(qi)
        if self._pend_move: self._upd_time()

    def _disp(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
        self.view.update_frame(qi); self._upd_time()

    def _upd_time(self):
        pr = self.cur_frame / max(1, self.total_frames)
        dur = self.total_frames / max(1, self.fps)
        self.lbl_time.setText(f"{_fmt(dur*pr)} / {_fmt(dur)}")
        self.slider.blockSignals(True); self.slider.set_position(pr); self.slider.blockSignals(False)
        self._upd_info()

    def _upd_info(self):
        if self.crop_w: self.lbl_coords.setText(f"裁剪 {self.crop_w}×{self.crop_h} @({self.crop_x},{self.crop_y})")
        else: self.lbl_coords.setText("拖拽画面选择裁剪区域")
        if self._range_e > self._range_s + 0.01 and self.total_frames:
            dur = self.total_frames / max(1, self.fps)
            rs = int(dur * self._range_s); re = int(dur * self._range_e)
            sel = dur * (self._range_e - self._range_s)
            if not self.edit_in.hasFocus(): self.edit_in.setTimeCode(rs)
            if not self.edit_out.hasFocus(): self.edit_out.setTimeCode(re)
            self.lbl_sel.setText(_fmt(sel))
        else:
            if not self.edit_in.hasFocus(): self.edit_in.setTimeCode(0)
            if not self.edit_out.hasFocus(): self.edit_out.setTimeCode(0)
            self.lbl_sel.setText("--:--")

    # ── 播放 ──
    def _toggle_play(self):
        if not self.cap: return
        if self._playing: self._stop()
        else: self._play()
    def _play(self):
        if not self.cap: return
        sf = int(self._range_s * self.total_frames)
        ef = int(self._range_e * self.total_frames)
        if self.cur_frame < sf or self.cur_frame >= ef:
            self._show_imm(sf)
        self._playing = True
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._play_acc = 0.0
        if self._play_speed > 1:
            self._del_pf()
            self._play_timer.start(int(1000 / self.fps))
        else:
            if self._prefetcher is None:
                self._prefetcher = FramePrefetcher(self.cap, 8)
            intv = int(1000 / (self.fps * max(self._play_speed, 0.25)))
            self._play_timer.start(max(8, intv))
    def _stop(self):
        self._playing = False; self._play_timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    def _del_pf(self):
        if self._prefetcher: self._prefetcher.stop(); self._prefetcher = None
    def _play_tick(self):
        if not self.cap: self._stop(); return
        if self._play_speed > 1:
            self._play_acc += self._play_speed
            skip = int(self._play_acc); self._play_acc -= skip
            if skip <= 0: return
            self.cur_frame += skip
            mf = int(self._range_e * self.total_frames) - 1
            if self.cur_frame >= mf: self.cur_frame = int(self._range_s * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
            r, f = self.cap.read()
            if not r: self._stop(); return
            self._disp(f)
            return
        if not self._prefetcher: self._prefetcher = FramePrefetcher(self.cap, 8)
        f = self._prefetcher.get()
        if f is None: return
        if f is FramePrefetcher._STOP: self._stop(); self._del_pf(); return
        self.cur_frame += 1
        mf = int(self._range_e * self.total_frames) - 1
        if self.cur_frame >= mf:
            self.cur_frame = int(self._range_s * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
            self._del_pf(); self._prefetcher = FramePrefetcher(self.cap, 8)
        self._disp(f)

    # ── 步骤 / 速度 ──
    def _step(self, d):
        if not self.cap: return
        self._stop(); self._show_imm(self.cur_frame+d)
    def _on_spd(self, i):
        self._play_speed = self._SPDS[i][1]
        if self._playing: self._stop(); self._play()
    def _adj_spd(self, d):
        if self._exporting: return
        c = self.combo_spd.currentIndex(); n = max(0, min(c+d, len(self._SPDS)-1))
        if n != c: self.combo_spd.setCurrentIndex(n)

    # ── 裁剪 ──
    def _on_crop(self, x, y, w, h):
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = x, y, w, h; self._upd_info()
    def _on_slider(self, s, e):
        self._stop()
        os, oe = self._range_s, self._range_e
        self._range_s, self._range_e = s, e; self._upd_info()
        if self.cap:
            if abs(e-oe) > abs(s-os): self._show_nopf(int(e*self.total_frames))
            else: self._show_nopf(int(s*self.total_frames))

    def _apply_in(self, sec=None):
        if not self.total_frames: return
        if sec is None: sec = self.edit_in.timeCode()
        dur = self.total_frames / max(1, self.fps)
        r = max(0, min(sec/dur, self._range_e-0.01))
        self._range_s = r; self.slider.set_range(r, self._range_e); self._on_slider(r, self._range_e)
    def _apply_out(self, sec=None):
        if not self.total_frames: return
        if sec is None: sec = self.edit_out.timeCode()
        dur = self.total_frames / max(1, self.fps)
        r = max(self._range_s+0.01, min(sec/dur, 1))
        self._range_e = r; self.slider.set_range(self._range_s, r); self._on_slider(self._range_s, r)

    # ── 预设 ──
    def _ref_presets(self):
        self.combo_preset.blockSignals(True); self.combo_preset.clear(); self.combo_preset.addItem("（不限）")
        if self.img_w > 0 and self.img_h > 0:
            for n,pw,ph in STANDARD_SIZES:
                if pw <= self.img_w and ph <= self.img_h: self.combo_preset.addItem(f"{n} ({pw}×{ph})", (pw,ph))
        self.combo_preset.blockSignals(False)
    def _on_preset(self, i):
        if i <= 0: return
        d = self.combo_preset.itemData(i)
        if d and self.img_w > 0:
            pw,ph = d; self.view.set_crop_exact((self.img_w-pw)//2, (self.img_h-ph)//2, pw, ph)

    # ── 导出 ──
    def _do_crop(self):
        if self._exporting: self._cancel_exp(); return
        if self.crop_w <= 0 or self.crop_h <= 0: QMessageBox.warning(self, "提示", "请先选择裁剪区域"); return
        if self.total_frames <= 0 or not self.video_path: return
        if self._export_thread and self._export_thread.is_alive(): return
        enc = self._get_enc()
        src = Path(self.video_path); sfx = "_crop"
        dur = self.total_frames / max(1, self.fps)
        ss = dur * self._range_s; to = dur * self._range_e
        sd = SEGMENT_OPTIONS[self.combo_seg.currentIndex()][1]; self._multi_segment = sd > 0
        if self._multi_segment:
            fo = src.parent / f"{src.stem}{sfx}_part01.mp4"
            if fo.exists() and QMessageBox.question(self, "覆盖？", f"{fo.name} 已存在，覆盖？", QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return
            op = str(src.parent / f"{src.stem}{sfx}")
        else:
            o = src.parent / f"{src.stem}{sfx}.mp4"
            if o.exists() and QMessageBox.question(self, "覆盖？", f"{o.name} 已存在，覆盖？", QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return
            op = str(o)
        self._set_lock(True); self._exporting = True; self._cancel_req = False; self._export_cur_sec = 0
        self.btn_crop.setText("✕取消"); self.btn_crop.setEnabled(True)
        try: self.btn_crop.clicked.disconnect()
        except (TypeError, RuntimeError): pass
        self.btn_crop.clicked.connect(self._cancel_exp)
        self.progress.setVisible(True); self.progress.setValue(0)
        if self._multi_segment:
            self.sig_status.emit("正在检测场景…")
            self._export_thread = threading.Thread(target=self._run_multi, args=(str(src),op,enc,ss,to,sd), daemon=True); self._export_thread.start()
        else:
            self.sig_status.emit(f"正在导出（{enc}）…")
            self._export_thread = threading.Thread(target=self._run_ff, args=(str(src),op,enc,ss,to), daemon=True); self._export_thread.start()

    def _build_cmd(self, src, out, enc, ss=0, to=0, accurate=False, crop=None):
        is_nvenc = enc.endswith("_nvenc"); cmd = [FFMPEG, "-y"]
        if ss > 0 and not accurate: cmd += ["-ss", f"{ss:.3f}"]
        cmd += ["-i", f"file:{src}"]
        if ss > 0 and accurate: cmd += ["-ss", f"{ss:.3f}"]
        if to > ss: cmd += ["-t", f"{to-ss:.3f}"]
        if crop: cx, cy, cw, ch = crop
        else: cw = self.crop_w; ch = self.crop_h; cx, cy = self.crop_x, self.crop_y
        if is_nvenc: cw &= ~1; ch &= ~1; cx &= ~1; cy &= ~1
        cmd += ["-vf", f"crop={cw}:{ch}:{cx}:{cy}"]
        if enc=="av1_nvenc": cmd += ["-c:v","av1_nvenc","-preset","p1","-cq","30","-b:v","0"]
        elif enc=="hevc_nvenc": cmd += ["-c:v","hevc_nvenc","-preset","p1","-cq","20","-b:v","0"]
        elif enc=="h264_nvenc": cmd += ["-c:v","h264_nvenc","-preset","p1","-cq","20","-b:v","0"]
        else: cmd += ["-c:v","libx264","-preset","fast","-crf","20","-threads","0"]
        cmd += ["-c:a","copy","-movflags","+faststart", out]; return cmd

    def _exec_ff(self, src, out, enc, ss=0, to=0, cb=None, accurate=False, crop=None):
        cmd = self._build_cmd(src, out, enc, ss, to, accurate, crop)
        lines = []
        p = None
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=1, text=True, errors='replace', creationflags=_CREATION_FLAGS)
            with self._lock: self._ffmpeg_proc = p
            dur = max(0.1, to-ss) if to>ss else 0
            tp = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
            for l in p.stderr:
                if not l: continue; lines.append(l)
                m = tp.search(l)
                if m:
                    h,m_,s = int(m.group(1)),int(m.group(2)),float(m.group(3))
                    cur = h*3600 + m_*60 + s
                    if dur > 0 and cb: cb(min(99, int(cur/dur*100)))
                    self._export_cur_sec = cur
            p.wait()
            with self._lock: self._ffmpeg_proc = None
            if p.returncode == 0: return True
            if p.returncode < 0: return False
            tail = "".join(lines[-20:]); self.sig_error.emit(f"导出失败（{p.returncode}）:\n{tail[-800:]}"); return False
        except Exception as e:
            with self._lock: self._ffmpeg_proc = None
            self.sig_error.emit(str(e)); return False

    def _run_ff(self, src, out, enc, ss=0, to=0):
        ok = self._exec_ff(src, out, enc, ss, to, cb=lambda p: self.sig_progress.emit(p))
        if ok: self.sig_progress.emit(100); self.sig_done.emit(out)
        self.sig_finish_exp.emit()

    def _run_multi(self, src, op, enc, ss, to, sd):
        try:
            scenes = self._detect_scenes(src, ss, to, lambda: self._cancel_req, lambda p: self.sig_progress.emit(int(p*.3)))
            with self._lock:
                if self._cancel_req: self.sig_status.emit("已取消"); return
            segs = self._comp_segs(scenes, ss, to, sd)
            n = len(segs); self._multi_msg = f"检测到 {len(scenes)} 场景，{n} 段"; self.sig_status.emit(self._multi_msg)
            okc = 0
            for i,(s,t) in enumerate(segs):
                with self._lock:
                    if self._cancel_req: break
                so = f"{op}_part{i+1:02d}.mp4"
                self._multi_msg = f"导出 {i+1}/{n}  ({_fmt(s)}→{_fmt(t)})"; self.sig_status.emit(self._multi_msg)
                bp = 30 + int(i/n*70); np2 = 30 + int((i+1)/n*70)
                if self._exec_ff(src, so, enc, s, t, lambda p,b=bp,np=np2: self.sig_progress.emit(b+int(p/100*(np-b))), accurate=True): okc += 1
                elif self._cancel_req: break
                else: self.sig_status.emit(f"分段 {i+1} 失败")
            if self._cancel_req: self.sig_status.emit(f"已取消（{okc}/{n}段）")
            else: self.sig_progress.emit(100); self.sig_done_msg.emit(f"完成：{okc}/{n}段 → {op}_partXX.mp4")
        except Exception as e: self.sig_error.emit(str(e))
        finally: self.sig_finish_exp.emit()

    def _detect_scenes(self, path, ss, to, cancel_cb, prog_cb):
        scenes = []; cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return scenes
        fps = cap.get(cv2.CAP_PROP_FPS) or 25; step = max(1, int(fps/2))
        sf = int(ss*fps); ef = int(to*fps); cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ph = None; cf = sf; lp = 0
        try:
            while cf < ef:
                if cancel_cb(): break
                r, f = cap.read(); cf += step
                if not r: break
                sm = cv2.resize(f, (160,90), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv],[0,1],None,[50,60],[0,180,0,256]); cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
                if ph is not None and cv2.compareHist(ph,hist,cv2.HISTCMP_BHATTACHARYYA) > 0.35:
                    ts = cf/fps
                    if not scenes or ts-scenes[-1] > 0.5: scenes.append(ts)
                ph = hist
                if cf < ef: cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
                pct = min(99, int((cf-sf)/max(1,ef-sf)*100))
                if pct-lp >= 2: lp = pct; prog_cb(pct)
        finally: cap.release()
        prog_cb(100); return scenes

    def _comp_segs(self, scenes, ss, to, td):
        total = to-ss
        if total <= td: return [(ss,to)]
        segs = []; cur = ss; mt = td*0.1
        while cur < to:
            tgt = cur+td
            if tgt >= to: segs.append((cur,to)); break
            best, bd = None, float('inf')
            smin, smax = cur+td*.5, cur+td*1.5
            for s in scenes:
                if s <= smin or s >= smax: continue
                d = abs(s-tgt)
                if d < bd: best = s; bd = d
            segs.append((cur, best or tgt)); cur = best or tgt
        if len(segs) >= 2:
            ls,le = segs[-1]
            if le-ls < mt: segs[-2] = (segs[-2][0], le); segs.pop()
        return segs

    def _cancel_exp(self):
        with self._lock: self._cancel_req = True; proc = self._ffmpeg_proc
        if proc and proc.poll() is None: proc.terminate()

    def _finish_exp(self):
        with self._lock: self._ffmpeg_proc = None
        self.progress.setVisible(False)
        self._restore_btn()
        self.view.clear_crop()
        title = self.windowTitle()
        if "已完成" not in title:
            self.setWindowTitle(f"{title}  ✅已完成")

    def _on_export_done(self, path):
        self.status.setStyleSheet("color:#00e676;font-weight:bold;")
        self.status.showMessage(f"导出完成：{Path(path).name}")

    def _on_export_done_msg(self, msg):
        self.status.setStyleSheet("color:#00e676;font-weight:bold;")
        self.status.showMessage(msg)

    def _on_status(self, m):
        self.status.setStyleSheet("")
        self.status.showMessage(m)

    def _restore_btn(self):
        with self._lock:
            if not self._exporting: return
            self._exporting = False
        try: self.btn_crop.clicked.disconnect()
        except (TypeError, RuntimeError): pass
        is_batch = self.batch_panel.isVisible()
        self.btn_crop.setText("✂导出预览视频" if is_batch else "✂导出裁剪")
        self.btn_crop.setEnabled(bool(FFMPEG))
        self.btn_crop.clicked.connect(self._do_crop); self._set_lock(False)

    def _set_lock(self, lk):
        for w in (self.btn_open, self.btn_batch, self.btn_batch_apply, self.combo_ratio, self.combo_preset, self.btn_clear,
                  self.combo_enc, self.combo_seg, self.edit_in, self.edit_out, self.combo_spd, self.slider):
            w.setEnabled(not lk)
        self.view._interact = not lk

    # ── 关闭 ──
    def closeEvent(self, e):
        with self._lock: self._cancel_req = True; proc = self._ffmpeg_proc
        self._stop(); self._del_pf()
        if proc and proc.poll() is None: proc.terminate()
        if self._export_thread and self._export_thread.is_alive(): self._export_thread.join(timeout=10)
        if self.cap: self.cap.release()
        super().closeEvent(e)


# ── 入口 ──
if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    w = MainWindow(); w.show(); sys.exit(app.exec())
