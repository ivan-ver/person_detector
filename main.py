from collections import OrderedDict

import cv2
import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.start_coords = None
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = maxDisappeared

    def register(self, centroid):
        self.objects[centroid[5]] = centroid
        self.disappeared[centroid[5]] = 0

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


def show_video():
    border_y = 730
    input_total = set()
    output_total = set()
    tracker = CentroidTracker()
    net = cv2.dnn.readNet('weights/yolov3.weights', 'weights/yolov3.cfg')
    cap = cv2.VideoCapture("video_2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_stream = cv2.VideoWriter('output.avi', fourcc, 30, (1920, 1080))

    while True:
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

            net.setInput(cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False))
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            Width = frame.shape[1]
            Height = frame.shape[0]

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
            objs = []

            for i in indices:
                i = i[0]
                box = boxes[i]
                if class_ids[i] == 0:
                    objs.append((box[0], box[1], box[0] + box[2], box[1] + box[3]))

            objects = tracker.update(objs)

            for i, (object_id, centroid) in enumerate(objects.items()):
                if (object_id > border_y) and (centroid[5] < border_y):
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[2] + 55), round(centroid[3] - 20)), (0, 0, 255), - 1)
                    cv2.putText(frame, f"ID {format(object_id)}", (centroid[2], centroid[3] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[4]), round(centroid[5])),
                                  (0, 0, 255), 1)
                    input_total.add(object_id)
                elif (object_id < border_y) and (centroid[5] > border_y):
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[2] + 55), round(centroid[3] - 20)), (0, 255, 0), - 1)
                    cv2.putText(frame, f"ID {format(object_id)}", (centroid[2], centroid[3] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[4]), round(centroid[5])),
                                  (0, 255, 0), 1)
                    output_total.add(object_id)
                else:
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[2] + 55), round(centroid[3] - 20)), (255, 0, 0), - 1)
                    cv2.putText(frame, f"ID {format(object_id)}", (centroid[2], centroid[3] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (round(centroid[2]), round(centroid[3])),
                                  (round(centroid[4]), round(centroid[5])),
                                  (255, 0, 0), 1)


            cv2.line(img=frame, pt1=(0, border_y), pt2=(1920, border_y), color=(125, 32, border_y), thickness=2)
            cv2.rectangle(frame, (0, 0), (200, 60), (0, 255, 0), - 1)
            cv2.putText(frame, f"INPUT: {len(input_total)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(frame, (0, 60), (200, 120), (0, 0, 255), - 1)
            cv2.putText(frame, f"OUTPUT: {len(output_total)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('frame', frame)
            out_stream.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    cap.release()
    out_stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_video()
