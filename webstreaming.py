from imutils.video import VideoStream
from imutils.video import FileVideoStream
from flask import Response
from flask import Flask, request
from flask import render_template
from darkflow.net.build import TFNet
import threading
import argparse
import datetime
import imutils
from sort import Sort
from utils import COLORS, intersect, get_output_fps_height_and_width
import numpy as np
import time
import cv2
import os
import math
import urllib.request
import youtube_dl
import pafy
import argparse   
from itertools import combinations

url = 'https://www.youtube.com/watch?v=rfkGy6dwWJs&feature=youtu.be'
vPafy = pafy.new(url)
play = vPafy.getbest()
WEBCAM_IP = 'http://192.168.43.1:8080/video'
path = '/home/suyash/Downloads/VideoDownloader/1.mp4'
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
#vs = VideoStream(src=0).start()
#vs = VideoStream(src=WEBCAM_IP).start()
#vs = VideoStream(src=play.url).start()
vs = FileVideoStream(path).start()
time.sleep(2.0)
tfnet = None
socialDistancing = None
delayTime = None


@app.route("/", methods=["GET", "POST"])
def home():
    if(request.method == 'POST'):
        global tfnet, socialDistancing, delayTime
        model = ''
        weights = ''
        name = request.form.get('name')
        delayTime = request.form.get('timeout')
        threshold = float(request.form.get('threshold'))
        yolov = request.form.get('yoloModel')
        if(yolov == 'yolov2-tiny-voc'):
            model = 'cfg/yolov2-tiny-voc.cfg'
            weights = 'bin/yolov2-tiny-voc.weights'
        if(yolov == 'yolov2'):
            model = 'cfg/yolov2-voc.cfg'
            weights = 'bin/yolov2-voc.weights'
        options = {"model": model,
                   "load": weights, "threshold": threshold, }
        sd = request.form.get('projectModel')
        if(sd == 'basic'):
            socialDistancing = False
        elif(sd == 'socialDistancing'):
            socialDistancing = True
        tfnet = TFNet(options)
        return render_template("setup.html", name=name, delayTime=delayTime, threshold=threshold, yolov=yolov, sd=sd)
    else:
        return render_template("starter.html")


@app.route("/feed")
def index():
    t.start()
    return render_template("index.html")


def detect_Persons(frameCount):
    global vs, outputFrame, lock, tfnet
    while True:
        frame = vs.read()
        frame1 = vs.read()
        objects = tfnet.return_predict(frame)
        if targeted_classes:
            objects = list(
                filter(lambda res: res["label"] in targeted_classes, objects))
        results, labels_quantities_dic = _convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(
            objects)
        _draw_detection_results(frame, results, labels_quantities_dic)
        _write_quantities(frame, labels_quantities_dic)
        frame = cv2.resize(frame, (900, 600),
                           interpolation=cv2.INTER_AREA)
        outputFrame = frame.copy()


def generate():
    global outputFrame, lock
    while True:
        if outputFrame is None:
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


def _convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(objects):
    labels_quantities_dic = {}
    results = []
    ids = 0

    for object in objects:
        x1, y1 = object["topleft"]["x"], object["topleft"]["y"]
        x2, y2 = object["bottomright"]["x"], object["bottomright"]["y"]
        confidence = object["confidence"]
        label = object["label"]

        try:
            labels_quantities_dic[label] += 1
        except KeyError:
            labels_quantities_dic[label] = 1

        start_point = (x1, y1)
        end_point = (x2, y2)
        mid_point = ((x1+x2)/2 , (y1+y2)/2)
        ids = ids + 1
        results.append((start_point, end_point, label, confidence, mid_point, ids))
    return results, labels_quantities_dic


def _draw_detection_results(frame, results, labels_quantities_dic):
    # Social distancing implementation
    if socialDistancing == True:
    	red_zone = []
    	for p1 , p2 in combinations(results, 2):
            dx = p1[4][0] - p2[4][0]
            dy = p1[4][1] - p2[4][1]
            if math.sqrt(dx**2 + dy**2) < 150:
                if p1[5] not in red_zone:
                    red_zone.append(p1[5])       #  Add Id to a list
                
                if p2[5] not in red_zone:
                   red_zone.append(p2[5])		# Same for the second id 
                   
    	for p in results:
        	if p[5] in red_zone:
        		cv2.rectangle(frame, p[0], p[1],
                      (0, 0, 255), DETECTION_FRAME_THICKNESS)
        	if p[5] not in red_zone:
        		cv2.rectangle(frame, p[0], p[1],
                      (0, 255, 0), DETECTION_FRAME_THICKNESS)
          
        
    	counterl = len(red_zone)
    	cv2.putText(
            frame,
            "People at risk: " + str(counterl),
            (10, 60),
            OBJECTS_ON_FRAME_COUNTER_FONT,
            OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
            (0, 0, 255),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,)
    else:
    	for p in results:
    		cv2.rectangle(frame, p[0], p[1],
                      (0, 255, 0), DETECTION_FRAME_THICKNESS) 
        
    # Exporting data to cloud    
    if int(time.time())%int(delayTime) <= 1:
            for i, (label, quantity) in enumerate(labels_quantities_dic.items()):    
                if socialDistancing == True:
                	people_at_danger = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=HV5A6SVYIMAGL9IK&field2="+str(counterl))
                total_people = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=HV5A6SVYIMAGL9IK&field1="+str(quantity)) 

def _write_quantities(frame, labels_quantities_dic):
    for i, (label, quantity) in enumerate(labels_quantities_dic.items()):
        class_id = [i for i, x in enumerate(
            labels_quantities_dic.keys()) if x == label][0]

        cv2.putText(
            frame,
            f"Total People: {quantity}",
            (10, (i + 1) * 35),
            OBJECTS_ON_FRAME_COUNTER_FONT,
            OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
            (255, 0, 0),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    DETECTION_FRAME_THICKNESS = 1
    OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
    OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 1
    LINE_COLOR = (0, 0, 255)
    LINE_THICKNESS = 3
    LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
    LINE_COUNTER_FONT_SIZE = 2.0
    LINE_COUNTER_POSITION = (20, 45)
    targeted_classes = ['person']
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    t = threading.Thread(target=detect_Persons, args=(
        args["frame_count"],))
    t.daemon = True
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
vs.stop()
