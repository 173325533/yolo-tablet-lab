# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
 YOLO 平板-电脑协同学习平台 —— 服务端主程序
═══════════════════════════════════════════════════════════════════

【整体架构】
  平板(浏览器,只负责显示和交互)
      │  同一 WiFi 局域网
      ▼
  这台电脑运行本程序(Flask Web 服务 + YOLO 推理/训练)

【请求处理流程】
  1. 平板访问 http://<电脑IP>:8000/          → 返回首页(推理演示)
  2. 平板访问 /annotate                       → 返回标注页面
  3. 平板访问 /train                          → 返回训练页面
  4. 图片检测:浏览器 JS 调 /detect           → 本程序调 YOLO → 返回画好框的图
  5. 实时画面:浏览器 <img> 连 /video         → 持续推送摄像头检测帧(MJPEG 流)
  6. 标注:浏览器 JS 调 /api/upload /api/label → 存入 dataset/ 目录
  7. 训练:浏览器 JS 调 /api/train/start      → 后台线程跑 YOLO 训练,
     浏览器轮询 /api/train/status 显示进度

【目录结构】
  yolo-web/
  ├── app.py            ← 本文件
  ├── templates/        ← 三个 HTML 页面(Flask 会自动在这里找)
  ├── static/uploads/   (预留)
  ├── models/           ← 训练产出的自定义权重 custom.pt
  └── dataset/          ← 标注数据集
      ├── images/       原始上传图片
      ├── labels/       每张图的 YOLO 格式标注(同名 .txt)
      ├── classes.json  类别名单(顺序即编号)
      ├── data.yaml     训练时自动生成的 ultralytics 配置
      ├── train/        训练时自动划分:训练集(80%)
      └── val/          训练时自动划分:验证集(20%)

【YOLO 标注格式说明】(labels/*.txt 每行一个框)
  类别编号 中心x 中心y 宽 高
  所有坐标都是 0~1 的比例值(除以图片宽高),与图片尺寸无关。
  例:0 0.250000 0.400000 0.300000 0.400000
  = 0 号类别的框,中心在图宽 25%、图高 40% 处,宽 30%、高 40%。
"""
import base64
import json
import os
import random
import re
import threading
import time

import cv2
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# ────────────────────────────────────────────────
# 一、目录初始化:把所有要用的文件夹建好
# ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")   # 上传的原始图片
LABELS_DIR = os.path.join(BASE_DIR, "dataset", "labels")    # YOLO 标注文件
MODELS_DIR = os.path.join(BASE_DIR, "models")               # 训练出的权重
for d in (UPLOAD_DIR, DATASET_DIR, LABELS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

CLASSES_FILE = os.path.join(BASE_DIR, "dataset", "classes.json")  # 类别名单
CUSTOM_WEIGHTS = os.path.join(MODELS_DIR, "custom.pt")       # 训练完成后的最优权重

app = Flask(__name__)

# ────────────────────────────────────────────────
# 二、加载模型
# ────────────────────────────────────────────────
# 当前用于检测的模型。启动时加载预训练的 yolo11n(认识 COCO 80 类物体);
# 训练完成后 run_training() 会把它换成你的自定义模型 custom.pt。
model = YOLO("yolo11n.pt")
model_lock = threading.Lock()  # 推理和"换模型"可能同时发生,加锁防止读到半初始化的模型


def load_classes():
    """读取类别名单。返回如 ["cup", "phone"],顺序就是类别编号 0、1。"""
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_classes(cls):
    """保存类别名单到 dataset/classes.json。"""
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(cls, f, ensure_ascii=False)


# ────────────────────────────────────────────────
# 三、训练状态
# ────────────────────────────────────────────────
# 训练跑在后台线程里,浏览器无法直接看到它,所以用这个全局字典共享进度:
# 训练线程负责写入,网页轮询接口 /api/train/status 负责读出。
train_state = {
    "running": False,   # 是否正在训练
    "done": False,      # 最近一次训练是否成功完成
    "progress": 0.0,    # 总进度 0~1
    "epoch": 0,         # 当前轮数
    "epochs": 0,        # 总轮数
    "log": "",          # 最近 40 行日志,展示在训练页面
    "map50": 0.0,       # 验证集 mAP50(检测准确率指标,越接近 1 越好)
}


def _log(msg):
    """往训练日志里追加内容,只保留最近 40 行,防止无限增长。"""
    train_state["log"] = (train_state["log"] + msg).splitlines()[-40:]
    train_state["log"] = "\n".join(train_state["log"])


def run_training(epochs, imgsz):
    """
    训练主流程(在后台线程中运行,不阻塞 Web 服务):

    第 1 步 划分数据集:把已标注的图片按 8:2 随机分成 train/val,
            生成 ultralytics 需要的 data.yaml 配置文件
    第 2 步 训练:基于预训练 yolo11n 做"迁移学习"(不是从零学,
            所以几十张图、几十轮就能出效果);每轮结束通过回调更新进度
    第 3 步 收尾:拷出最优权重 best.pt → custom.pt,
            并把首页检测模型切换为它
    """
    global model
    try:
        train_state.update(running=True, done=False, progress=0.0, log="")

        # ---- 第 1 步:准备数据集 ----
        all_imgs = [f for f in os.listdir(DATASET_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # 只有存在对应 .txt 标注的图片才能参与训练
        labeled = [f for f in all_imgs
                   if os.path.exists(os.path.join(LABELS_DIR, os.path.splitext(f)[0] + ".txt"))]
        if len(labeled) < 10:
            _log(f"已标注图片太少({len(labeled)} 张,至少 10 张),中止。\n")
            return

        # 随机打乱后 8:2 分割(前 20% 做验证,后 80% 做训练)
        random.shuffle(labeled)
        n_val = max(1, len(labeled) // 5)
        val_set, train_set = labeled[:n_val], labeled[n_val:]

        # 用硬链接把图片"复制"到 train/ val/ 子目录(硬链接不占额外磁盘空间)
        for sub in ("train", "val"):
            for d in (os.path.join(BASE_DIR, "dataset", sub, "images"),
                      os.path.join(BASE_DIR, "dataset", sub, "labels")):
                os.makedirs(d, exist_ok=True)
        for f in labeled:
            sub = "val" if f in val_set else "train"
            dst_img = os.path.join(BASE_DIR, "dataset", sub, "images", f)
            dst_lbl = os.path.join(BASE_DIR, "dataset", sub, "labels",
                                   os.path.splitext(f)[0] + ".txt")
            src_img = os.path.join(DATASET_DIR, f)
            src_lbl = os.path.join(LABELS_DIR, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(dst_img):
                os.link(src_img, dst_img)
            if not os.path.exists(dst_lbl) and os.path.exists(src_lbl):
                open(dst_lbl, "w").write(open(src_lbl).read())

        # 生成 ultralytics 的数据集配置文件
        data_yaml = os.path.join(BASE_DIR, "dataset", "data.yaml")
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write(f"path: {BASE_DIR}\\dataset\ntrain: train/images\nval: val/images\n"
                    f"nc: {len(load_classes())}\nnames: {load_classes()}\n")
        _log(f"训练集 {len(train_set)} 张,验证集 {len(val_set)} 张\n")

        # ---- 第 2 步:训练 ----
        # on_epoch 回调:每轮结束被 ultralytics 调用,把进度写进 train_state,
        # 这样平板页面每 3 秒轮询一次就能看到实时进度
        def on_epoch(trainer):
            train_state["epoch"] = trainer.epoch + 1
            train_state["epochs"] = trainer.epochs
            train_state["progress"] = (trainer.epoch + 1) / trainer.epochs
            m = trainer.metrics or {}
            train_state["map50"] = float(m.get("metrics/mAP50(B)", 0))
            _log(f"epoch {trainer.epoch+1}/{trainer.epochs}  "
                 f"loss={float(m.get('train/box_loss', 0)):.3f}  "
                 f"mAP50={train_state['map50']:.3f}\n")

        # 重新加载一份干净的预训练模型来训练(不动推理用的 model)
        # device="cpu"    在 CPU 上训练(GPU 电脑可改成 "0")
        # batch=8         每批 8 张图一起算,CPU 内存友好
        # workers=0       Windows 下多进程加载容易出问题,单线程最稳
        m2 = YOLO("yolo11n.pt")
        m2.add_callback("on_fit_epoch_end", on_epoch)
        results = m2.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                           device="cpu", batch=8, workers=0, exist_ok=True,
                           project=MODELS_DIR, name="run")

        # ---- 第 3 步:收尾 ----
        best = os.path.join(MODELS_DIR, "run", "weights", "best.pt")
        if os.path.exists(best):
            import shutil
            shutil.copy(best, CUSTOM_WEIGHTS)
            with model_lock:               # 加锁换模型,正在进行的推理不受影响
                model = YOLO(CUSTOM_WEIGHTS)
            train_state["done"] = True
            _log("训练完成,已切换到自定义模型!✅\n")
    except Exception as e:
        _log(f"训练出错: {e}\n")
    finally:
        train_state["running"] = False


# ────────────────────────────────────────────────
# 四、检测核心函数(图片检测和摄像头共用)
# ────────────────────────────────────────────────
def draw_detections(frame):
    """
    输入一帧图像 → YOLO 检测 → 画出框。
    返回 (画好框的图, 检测结果列表)。
    结果列表形如 [{"class": "cup", "conf": 0.92}, ...],用于网页上显示文字标签。
    """
    with model_lock:
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
    annotated = results[0].plot()          # ultralytics 自带画框(框+类别+置信度)
    detected = []
    for box in results[0].boxes:
        detected.append({
            "class": model.names[int(box.cls)],   # 类别名(如 "person")
            "conf": round(float(box.conf), 3),    # 置信度 0~1
        })
    return annotated, detected


# ────────────────────────────────────────────────
# 五、页面路由(返回 HTML)
# ────────────────────────────────────────────────
@app.route("/")
def index():
    """首页:图片检测 + 摄像头实时检测 + YOLO 知识讲解。"""
    return render_template("index.html")


@app.route("/annotate")
def annotate_page():
    """标注页:上传图片、画框、保存 YOLO 格式标注。"""
    return render_template("annotate.html")


@app.route("/train")
def train_page():
    """训练页:设置参数、启动训练、查看进度和结果。"""
    return render_template("train.html")


# ────────────────────────────────────────────────
# 六、图片检测接口(首页「开始检测」按钮调用)
# ────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """接收平板上传的一张图片 → 检测 → 返回 JSON(base64 图片 + 物体列表 + 耗时)。"""
    if "image" not in request.files:
        return jsonify({"error": "没有收到图片"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400

    # 平板拍的照片可能有几 MB,直接推理会很慢:
    # 先解码成内存中的图像数组,长边压到 960 像素再检测
    import numpy as np
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "无法解码图片"}), 400
    h, w = img.shape[:2]
    scale = 960 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    t0 = time.time()
    annotated, detected = draw_detections(img)
    cost = round(time.time() - t0, 2)

    # 把结果图编码成 JPEG 再转 base64,直接塞进 JSON 返回;
    # 浏览器端 <img src="data:image/jpeg;base64,..."> 即可显示,无需存文件
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return jsonify({
        "image": "data:image/jpeg;base64," + b64,
        "objects": detected,
        "seconds": cost,
    })


# ────────────────────────────────────────────────
# 七、摄像头实时流(首页「打开实时画面」使用)
# ────────────────────────────────────────────────
def generate_frames():
    """
    打开电脑摄像头,循环:取一帧 → 检测画框 → 压成 JPEG → 推给浏览器。
    这是 MJPEG 流:服务端不停地发 "--frame<JPEG>--frame<JPEG>...",
    浏览器的 <img> 标签会自动连续渲染,看起来就是视频。
    CPU 推理约每秒几帧,足够看清检测效果。
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DMSHOW if False else cv2.CAP_DSHOW)  # Windows 指定 DirectShow 后端,打开更快
    if not cap.isOpened():
        # 没有摄像头时返回一张写着提示的黑色图片,而不是报错
        placeholder = "Camera not available on PC"
        frame = cv2.putText(cv2.Mat.zeros((480, 640, 3), cv2.uint8), placeholder,
                            (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 480))
            annotated, _ = draw_detections(frame)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()  # 无论怎么退出都释放摄像头,否则其他程序打不开它


@app.route("/video")
def video():
    """MJPEG 流式响应。浏览器收到后持续渲染,连接断开时生成器自动终止。"""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ────────────────────────────────────────────────
# 八、标注相关接口(标注页的 JS 调用)
# ────────────────────────────────────────────────
@app.route("/dataset/<path:name>")
def dataset_file(name):
    """提供 dataset/images/ 里图片的访问(标注页缩略图和预览用)。"""
    return send_from_directory(DATASET_DIR, name)


@app.route("/api/images")
def api_images():
    """列出所有已上传图片及是否已标注(标注页启动时加载)。"""
    imgs = sorted(f for f in os.listdir(DATASET_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return jsonify({"images": [
        {"name": f,
         "labeled": os.path.exists(os.path.join(LABELS_DIR, os.path.splitext(f)[0] + ".txt"))}
        for f in imgs]})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """接收平板批量上传的图片,存到 dataset/images/。重名跳过防止覆盖。"""
    saved, skipped = 0, 0
    for f in request.files.getlist("files"):
        if not f.filename:
            continue
        dst = os.path.join(DATASET_DIR, f.filename)
        if os.path.exists(dst):
            skipped += 1
            continue
        f.save(dst)
        saved += 1
    return jsonify({"saved": saved, "skipped": skipped})


@app.route("/api/classes", methods=["GET", "POST"])
def api_classes():
    """GET 读取类别名单;POST 保存(标注页「保存类别」按钮)。"""
    if request.method == "POST":
        cls = request.get_json().get("classes", [])
        save_classes([str(c) for c in cls])
    return jsonify({"classes": load_classes()})


@app.route("/api/label", methods=["POST"])
def api_label():
    """
    保存一张图的标注框。
    前端传来归一化的左上角+宽高,这里转成 YOLO 要求的"中心点+宽高"格式写入 .txt。
    坐标换算:x_center = x + w/2 ,y_center = y + h/2。
    """
    d = request.get_json()
    name = os.path.basename(d["name"])  # 只取文件名,防止恶意路径(如 ../../)穿越
    boxes = d.get("boxes", [])
    lines = []
    for b in boxes:
        xc = b["x"] + b["w"] / 2
        yc = b["y"] + b["h"] / 2
        lines.append(f'{b["cls"]} {xc:.6f} {yc:.6f} {b["w"]:.6f} {b["h"]:.6f}')
    with open(os.path.join(LABELS_DIR, os.path.splitext(name)[0] + ".txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return jsonify({"ok": True, "boxes": len(lines)})


# ────────────────────────────────────────────────
# 九、训练相关接口(训练页的 JS 调用)
# ────────────────────────────────────────────────
@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    """启动训练。真正的训练跑在 daemon 线程里,本接口立即返回不阻塞。"""
    if train_state["running"]:
        return jsonify({"error": "训练正在进行中"}), 400
    p = request.get_json() or {}
    threading.Thread(target=run_training,
                     kwargs={"epochs": int(p.get("epochs", 30)),
                             "imgsz": int(p.get("imgsz", 480))},
                     daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    """CPU 训练中断比较复杂,这里引导用户直接重启服务。"""
    return jsonify({"error": "请直接关闭并重启 app.py 来停止训练(CPU 训练不会损坏已有数据)"}), 400


@app.route("/api/train/status")
def api_train_status():
    """
    训练页每 3~5 秒轮询一次本接口:
    返回数据集统计 + 训练进度 + 日志 + mAP,前端据此刷新进度条和图表。
    """
    imgs = [f for f in os.listdir(DATASET_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    labeled = [f for f in imgs
               if os.path.exists(os.path.join(LABELS_DIR, os.path.splitext(f)[0] + ".txt"))]
    return jsonify({
        "total": len(imgs), "labeled": len(labeled),
        "classes": load_classes(),
        "running": train_state["running"], "done": train_state["done"],
        "progress": train_state["progress"], "epoch": train_state["epoch"],
        "epochs": train_state["epochs"], "log": train_state["log"],
        "map50": train_state["map50"],
    })


@app.route("/api/train/curve")
def api_train_curve():
    """返回 ultralytics 自动绘制的训练曲线图 results.png。"""
    p = os.path.join(MODELS_DIR, "run", "results.png")
    if os.path.exists(p):
        return send_from_directory(os.path.join(MODELS_DIR, "run"), "results.png")
    return "", 404


# ────────────────────────────────────────────────
# 十、启动入口
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # host="0.0.0.0" 是关键:监听所有网卡,同一 WiFi 里的平板才能访问
    # threaded=True 让检测、看实时流、查训练进度可以同时进行
    app.run(host="0.0.0.0", port=8000, threaded=True)
