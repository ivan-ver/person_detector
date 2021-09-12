from collections import OrderedDict

import cv2
import numpy as np
from scipy.spatial import distance as dist


class Border:
    a_x = None
    a_y = None
    b_x = None
    b_y = None

    def __init__(self, a_x, a_y, b_x, b_y, video_path):
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        height, width, channels = frame.shape
        self.a_x = int(a_x / 100 * width)
        self.a_y = int(a_y / 100 * height)
        self.b_x = int(b_x / 100 * width)
        self.b_y = int(b_y / 100 * height)

    def check(self, x, y):
        return (self.b_x - self.a_x) * (y - self.a_y) - (self.b_y - self.a_y) * (x - self.a_x) > 0


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.start_coords = None
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = maxDisappeared

    def register(self, centroid):
        self.objects[str(centroid[2]) + "_" + str(centroid[3])] = centroid
        self.disappeared[str(centroid[2]) + "_" + str(centroid[3])] = 0

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        input_centroids = np.zeros((len(rects), 6), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y, start_x, start_y, end_x, end_y)
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects


def load_yolo():
    net = cv2.dnn.readNet('yolo_cfg/yolov3.weights', 'yolo_cfg/yolov3.cfg')
    with open("yolo_cfg/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.1 and class_id == 0:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
    return boxes, confs


def draw_labels(boxes, confs, img, tracker, input_total, output_total, border, out_stream):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.5)
    objects = list()
    for i in range(len(boxes)):
        if i in indexes:
            objects.append(boxes[i])
    objects = tracker.update(objects)
    for (object_id, centroid) in objects.items():

        if (border.check(x=int(object_id.split('_')[0]), y=int(object_id.split('_')[1]))) and (
                not border.check(x=centroid[2] + centroid[4], y=centroid[3] + centroid[5])):

            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + 65), round(centroid[3] - 20)), (0, 0, 255), - 1)
            cv2.putText(img, f"ID {format(object_id.split('_')[0])}", (centroid[2], centroid[3] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + centroid[4]), round(centroid[3] + centroid[5])), (0, 0, 255), 1)
            output_total.add(object_id)

        elif (not border.check(x=int(object_id.split('_')[0]), y=int(object_id.split('_')[1]))) and (
                border.check(x=centroid[2] + centroid[4], y=centroid[3] + centroid[5])):

            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + 65), round(centroid[3] - 20)), (0, 255, 0), - 1)
            cv2.putText(img, f"ID {format(object_id.split('_')[0])}", (centroid[2], centroid[3] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)
            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + centroid[4]), round(centroid[3] + centroid[5])), (0, 255, 0), 1)
            input_total.add(object_id)
        else:
            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + 65), round(centroid[3] - 20)), (255, 0, 0), - 1)
            cv2.putText(img, f"ID {format(object_id.split('_')[0])}", (centroid[2], centroid[3] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.rectangle(img, (round(centroid[2]), round(centroid[3])),
                          (round(centroid[2] + centroid[4]), round(centroid[3] + centroid[5])), (255, 0, 0), 1)

    cv2.line(img=img, pt1=(border.a_x, border.a_y), pt2=(border.b_x, border.b_y), color=(125, 32, 0), thickness=2)
    cv2.rectangle(img, (0, 0), (200, 60), (0, 255, 0), - 1)
    cv2.putText(img, f"INPUT: {len(input_total)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(img, (0, 60), (200, 120), (0, 0, 255), - 1)
    cv2.putText(img, f"OUTPUT: {len(output_total)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    out_stream.write(img)


def start_video(video_path, tracker, border):
    input_total = set()
    output_total = set()
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_stream = cv2.VideoWriter('output.avi', fourcc, 30, (1920, 1080))

    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs = get_box_dimensions(outputs, height, width)
        draw_labels(boxes=boxes, confs=confs, img=frame, tracker=tracker, output_total=output_total,
                    input_total=input_total, border=border, out_stream=out_stream)
        key = cv2.waitKey(1)
        if key == 27 or not ret:
            break

    cap.release()


if __name__ == '__main__':
    tracker = CentroidTracker()
    border = Border(a_x=0, a_y=250, b_x=1920, b_y=650)
    start_video('video_2.mp4', tracker=tracker, border=border)
