
# Multi-Camera Object Tracking  

Track and identify objects across multiple video feeds (or cameras).  
Logs detections with timestamps in a SQLite database and provides a simple Flask web dashboard for reviewing.  

---

## 📝 Description  
This project uses **YOLOv8** for object detection and **StrongSORT** for multi-object tracking.  
Objects are assigned **global IDs** and matched across cameras.  
All detections are saved in a SQLite database (`tracks.db`) for later analysis.  

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-orange?logo=github" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite" alt="SQLite">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker" alt="Docker">
</p>



## ⚡ Tech Used  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection  
- [StrongSORT](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet) – Multi-object tracking  
- [PyTorch](https://pytorch.org/) – Deep learning backbone  
- [OpenCV](https://opencv.org/) – Video processing & visualization  
- [Flask](https://flask.palletsprojects.com/) – Web dashboard  
- [SQLite + SQLAlchemy](https://www.sqlalchemy.org/) – Lightweight database for logs  

---

## 🔮 Future Updates  

- **Better Web App UI/UX:**  
  - Grid layout for multiple cameras  
  - Play/pause buttons for video feeds  
  - Real-time object count per camera  

- **Filter & Highlight Objects:**  
  - Highlight a specific global ID across cameras  
  - Color-coded bounding boxes per object  

- **Download / Export Logs:**  
  - Add a button to download CSV of detections  

- **Simple Dashboard Metrics:**  
  - Table or panel showing total objects detected per camera  
  - Last detection timestamps for quick reference  

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.