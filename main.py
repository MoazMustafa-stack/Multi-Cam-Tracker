import cv2
import yaml
import numpy as np
import queue

from datetime import datetime, timezone
from ultralytics import YOLO
from strongsort.strong_sort import StrongSort
from db import Session, Detection

frame_queue = {} # Frame queue for each camera

global_id_counter = 0 # Global ID counter
active_tracks = {} # Active tracks for each camera

# Cross match function
def cross_match(candidates, cosine_thresh=0.35):
    global global_id_counter, active_tracks
    mapping = {}

    for i, cand in enumerate(candidates):
        assigned = None
        for gid, prev in active_tracks.items():
            emb_c = cand.get("embedding")
            emb_p = prev.get("embedding")
            if emb_c is not None and emb_p is not None:
                cos = float(np.dot(emb_c, emb_p) / (np.linalg.norm(emb_c) * np.linalg.norm(emb_p) + 1e-6))
                if cos > cosine_thresh:
                    assigned = gid
                    active_tracks[gid] = cand
                    break

        if assigned is None:
            global_id_counter += 1
            assigned = global_id_counter
            active_tracks[assigned] = cand

        mapping[i] = assigned

    return mapping

# Run tracker function
def run_tracker():
    # Configurations
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    cameras = config.get("cameras", {})
    DEVICE = config.get("device", "cpu")
    CONF_THRES = float(config.get("confidence", 0.5))
    COSINE_THRES = float(config.get("cosine_threshold", 0.35))

    # Initialize models
    detector = YOLO(config.get("model", "yolov8n.pt"))

    trackers = {cam: StrongSort(max_age=30, n_init=3, nn_budget=100) for cam in cameras}
    caps = {cam: cv2.VideoCapture(src) for cam, src in cameras.items()}
    frame_counts = {cam: 0 for cam in cameras}
    session = Session()

    # Main loop
    while True:
        any_frame = False
        candidates = []
        frames = {}

        for cam, cap in caps.items():
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            any_frame = True
            frames[cam] = frame
            frame_counts[cam] += 1
            ts = datetime.now(timezone.utc)

            # Run detector
            results = detector(frame, conf=CONF_THRES, device=DEVICE, verbose=False)
            bboxes, confs = [], []

            rs = results[0]
            if rs.boxes is not None:
                for b in rs.boxes:
                    # person class id in COCO is 0
                    cls_id = int(b.cls[0].item()) if hasattr(b, 'cls') else 0
                    if cls_id == 0:
                        xyxy = b.xyxy[0].cpu().numpy().ravel()
                        conf = float(b.conf[0].cpu().item()) if hasattr(b, 'conf') else 1.0
                        bboxes.append(xyxy)
                        confs.append(conf)

            if len(bboxes) == 0:
                continue

            bboxes_np = np.array(bboxes, dtype=np.float32)
            confs_np = np.array(confs, dtype=np.float32)

            # Run tracker
            outputs = trackers[cam].update(bboxes_np, confs_np, frame)

            # Process outputs
            for out in outputs:
                # Expecting at least [x1,y1,x2,y2,track_id,...]
                if len(out) < 5:
                    continue
                x1, y1, x2, y2 = out[0], out[1], out[2], out[3]
                track_id = out[4]
                conf = float(out[6]) if len(out) > 6 else 1.0
                emb = out[7] if len(out) > 7 else None
                candidates.append({
                    "cam": cam,
                    "local_id": int(track_id),
                    "frame_idx": frame_counts[cam],
                    "timestamp": ts,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "conf": conf,
                    "embedding": emb,
                })
        
        if not any_frame:
            break
        if not candidates:
            # still update frame_queue with latest frames so UI shows raw streams
            for cam, frm in frames.items():
                if cam not in frame_queue:
                    frame_queue[cam] = queue.Queue(maxsize=1)
                if not frame_queue[cam].empty():
                    try:
                        frame_queue[cam].get_nowait()
                    except Exception:
                        pass
                try:
                    frame_queue[cam].put_nowait(frm)
                except Exception:
                    pass
            continue

        # Cross match   
        mapping = cross_match(candidates, COSINE_THRES)

        # Add detections to database and draw boxes
        for i, c in enumerate(candidates):
            gid = mapping[i]

            det = Detection(
                global_id=gid,
                camera_id=c["cam"],
                local_track_id=c["local_id"],
                frame_index=c["frame_idx"],
                timestamp=c["timestamp"],
                bbox_x1=c["bbox"][0],
                bbox_y1=c["bbox"][1],
                bbox_x2=c["bbox"][2],
                bbox_y2=c["bbox"][3],
                confidence=c["conf"],
            )
            session.add(det)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, c["bbox"])
            frm = frames.get(c["cam"]) 
            if frm is not None:
                cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frm, str(gid), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        session.commit()

        # Update frame queue
        for cam, frm in frames.items():
            if cam not in frame_queue:
                frame_queue[cam] = queue.Queue(maxsize=1)
            if not frame_queue[cam].empty():
                try:
                    frame_queue[cam].get_nowait()
                except Exception:
                    pass
            try:
                frame_queue[cam].put_nowait(frm)
            except Exception:
                pass


    for cap in caps.values():
        cap.release()
    session.close()
    cv2.destroyAllWindows()
    print("Tracking completed...")