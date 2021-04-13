from __future__ import print_function
import sys
import os, io
from argparse import ArgumentParser, SUPPRESS
import cv2, queue, threading
import time
import logging as log
import json
import requests
from predict import initialize, predict_image
from flask import jsonify, Flask, render_template, Response
from PIL import Image
import numpy as np
from itertools import count
import onnxruntime as rt
from onnxruntime import get_device

from openvino.inference_engine import IENetwork, IECore

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=False, type=str, default='face-detection-retail-0004.xml')
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=False, type=str, default='cam')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    return parser

def gen():
    sess = rt.InferenceSession('model.onnx')
    try:
        with open(r'/setting/settings.json', 'r') as f:
            data = json.load(f)
        localip = data['localip']
    except:
        localip = 'http://192.168.1.222'

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
            sys.exit(1)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    # bufferless VideoCapture
    class VideoCapture:
        def __init__(self, name):
            self.cap = cv2.VideoCapture(name)
            self.q = queue.Queue()
            t = threading.Thread(target=self._reader)
            t.daemon = True
            t.start()

        # read frames as soon as they are available, keeping only most recent one
        def _reader(self):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.q.put(frame)

        def read(self):
            return self.q.get()
    # cap = cv2.VideoCapture('http://192.168.1.66:8081/video.mjpg')
    cap = VideoCapture(localip + ':8877/video.mjpg')
    # assert cap.isOpened(), "Can't open " + input_stream

    if args.labels:
            with open(args.labels, 'r') as f:
                labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    is_async_mode = False
    render_time = 0
    if is_async_mode:
        frame = cap.read()
        # ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    ##Initiate counter for frame processing
    maskresults = []
    maskprobs = []
    boxes = []
    cnt = count(0)
    while True:
        try:
            with open(r'/setting/settings.json', 'r') as f:
                data = json.load(f)
            bbox_num = data['bbox_num']
            threshold = data['threshold']
            iter_per_inf = data['iter_per_inf']
        except:
            bbox_num = 20
            threshold = 0.1
            iter_per_inf = 3
        if is_async_mode:
            next_frame = cap.read()
            # ret, next_frame = cap.read()
        else:
            # ret, frame = cap.read()
            frame = cap.read()
            # if ret:
                # frame_h, frame_w = frame.shape[:2]
            frame_h, frame_w = frame.shape[:2]
            
        ##Tell Chaomin to grab
        try:
            r = requests.get(localip + ':8066/WebService/GrabTrigger?Grab=true')
        except:
            pass
        
        # if not ret:
            # break 
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            bbox = []
            maskresults = maskresults
            maskprobs = maskprobs
            boxes = boxes
            
            ## Start iterating through for loop of detected objects
            forcnt = count(0)
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    ## check if it's first for loop iteration
                    fidx = next(forcnt)
                    if fidx is 0:
                        idx = next(cnt)
                    else:
                        None
                    if idx % iter_per_inf is 0:
                        xmin = int(obj[3] * frame_w)
                        ymin = int(obj[4] * frame_h)
                        xmax = int(obj[5] * frame_w)
                        ymax = int(obj[6] * frame_h)
                        if fidx is 0:
                            maskresults = []
                            maskprobs = []
                            boxes = []
                        else:
                            pass
                        ## do classification inference every n frames
                        if frame.shape[0] == 0 or frame.shape[1] == 0:
                            maskprob = 0.00
                            masked = "Unknown"
                            imageshape = "not an image"
                        else:
                            #save cropped images for training
                            # cv2.imwrite(r'/train/' + str(idx) + ".jpg", im)
                            try:
                                im = frame[ymin:ymax, xmin:xmax]
                                # convert from RGB to BGR
                                # im = im[:, :, (2,1,0)]
                                im = cv2.resize(im, (224,224))
                                x = np.asarray(im).astype(np.float32)
                                x = np.transpose(x, (2,0,1))
                                x = np.expand_dims(x,axis=0)
                                try:
                                    res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
                                    masked = res[0][0][0]
                                    maskprob = 0.00
                                except:
                                    masked = "Unknown"
                                    maskprob = 0.00
                                imageshape = "an image"
                            except:
                                masked = "Unknown"
                                maskprob = 0.00
                                imageshape = "not an image"
                        box_array = [xmin, ymin, xmax, ymax]
                        boxes.append(box_array)
                        maskresults.append(masked)
                        maskprobs.append(maskprob)
                    else:
                        pass
                    try:
                        if maskresults[fidx] == "Facemask":
                            color = (255,0,0)
                        elif maskresults[fidx] == "Nomask":
                            color = (0,0,255)
                        else:
                            color = (0,255,255)
                        masked = maskresults[fidx]
                        maskprob = maskprobs[fidx]
                        xmin = boxes[fidx][0]
                        ymin = boxes[fidx][1]
                        xmax = boxes[fidx][2]
                        ymax = boxes[fidx][3]
                    except:
                        masked = "Unknown"
                        maskprob = 0.00
                        color = (0,255,255)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
                    # det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, masked, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    singlebox = {"Left":str(xmin),"Top":str(ymin),"Right":str(xmax),"Bottom":str(ymax)}
                    bbox.append(singlebox)
            if idx % iter_per_inf is 0:
                try:
                    if imageshape is "an image":
                        # print(maskresults)
                        ## Sending BoundingBoxes to ChaoMin####
                        jsondata = {"BoundingBox":bbox,"Mask":maskresults}
                        json_data = json.dumps(jsondata)
                        url = localip + ":8066/WebService/RecieveCoordinate"
                        r = requests.post(url = url, data = json_data)
                        POST_response = r.text
                        json_response = json.loads(POST_response)
                        returncode = json_response["ReturnCode"]
                        if returncode is "0":
                            temp_data = json_response["Data"]
                            max_temp_array = temp_data["MaxTemperature"]
                        else:
                            max_temp_array = []
                        #######################################
                    else:
                        pass
                except:
                    pass
            else:
                pass
                    
        render_start = time.time()
        # cv2.imshow("Detection Results", frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        htmlimage = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + htmlimage + b'\r\n\r\n')
        
        render_end = time.time()
        render_time = render_end - render_start
        
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
        # if (9 == key):
        #     is_async_mode = not is_async_mode
        #     log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # initialize()
    # sys.exit(main() or 0)
    app.run(host='0.0.0.0',port='8432', debug=True)