# YOLO 平板-电脑协同学习平台

电脑(CPU)运行 YOLO 推理服务,平板连同一 WiFi,用浏览器访问即可。

## 启动

```bash
cd yolo-web
python app.py
```

## 平板访问

平板浏览器打开:**http://192.168.2.57:8000**
(具体 IP 以 `ipconfig` 查询结果为准;若 Windows 防火墙拦截,首次运行时选择"允许访问")

## 功能

- 📸 平板拍照/选图上传 → 电脑 YOLO 检测 → 返回标注框图片 + 物体列表 + 耗时
- 🎥 电脑摄像头实时检测流(MJPEG),平板实时观看
- 📚 页面内置 YOLO 知识讲解和小实验建议

## 技术栈

Flask + ultralytics(yolo11n) + OpenCV,CPU 推理约 0.3s/张。

## 常改的参数(app.py)

- `conf=0.4` — 置信度阈值,调低识别更多但误报多
- `YOLO("yolo11n.pt")` — 可换 yolo11s.pt 更准但更慢
- `port=8000` — 服务端口
