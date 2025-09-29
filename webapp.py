from flask import Flask, render_template_string, Response
import cv2
from main import frame_queue, run_tracker
import threading

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Camera Tracker</title>
</head>
<h2>Live Multi-Camera Streams</h2>
{% for cam in cameras %}
    <img src="/video_feed/{{ cam }}" width="640">
{% endfor %}
</html>
"""

def generate_frames(cam):
    while True:
        if cam in frame_queue and not frame_queue[cam].empty():
            frame = frame_queue[cam].get()
            ret,buffer = cv2.imencode('.jpg',frame)
            if not ret:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, cameras=frame_queue.keys())

@app.route('/video_feed/<cam>')
def video_feed(cam):
    return Response(generate_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    thread = threading.Thread(target=run_tracker, daemon=True)
    thread.start()
    app.run(host='0.0.0.0',port=5000,threaded=True)
