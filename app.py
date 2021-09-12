import logging

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from main import CentroidTracker, start_video, Border

app = Flask(__name__)
app.logger.disabled = True
socketIO = SocketIO(app, logger=False, engineio_logger=False)

log = logging.getLogger('werkzeug')
log.disabled = True
import asyncio

tracker = CentroidTracker()

robot_event_loop = asyncio.new_event_loop()


@app.route('/')
def start_app():
    return render_template('index.html')


@socketIO.on('on_connect')
def on_connect(data):
    emit("message", "CONNECT")
    print("CONNECT")


@socketIO.on('load_data')
def handle_message(data):
    border = Border(a_x=int(data['x1']), a_y=int(data['y1']),
                    b_x=int(data['x2']), b_y=int(data['y2']),
                    video_path=data['path'])
    start_video(video_path=data['path'], tracker=tracker, border=border)


if __name__ == '__main__':
    socketIO.run(app)
