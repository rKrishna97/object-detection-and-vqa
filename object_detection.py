import numpy as np
import cv2
import time
import os

from numpy.lib.type_check import imag 

class ObjectDetection:
    def __init__(self, video_filename = "video", image_filename = "image"):
        self.video_filename = video_filename
        self.image_filename = image_filename
        self.video_folder = "./static/video/"
        self.video_file_path = os.path.join(self.video_folder, video_filename)
        self.image_folder = "./static/image/"
        self.image_file_path = os.path.join(self.image_folder, image_filename)

        


        

    def getOutputFrame(self,frame):
        

        h,w = None, None

        if w is None or h is None:
            h, w = frame.shape[:2]

        frame_center = w/2
        pos_thres = int(0.2 * w)
        lb = frame_center - pos_thres
        ub = frame_center + pos_thres
        

        with open("./yolo_model/coco.names") as f:
            labels = [line.strip() for line in f]

        if self.image_filename != "image" and self.video_filename == "video":
            network = cv2.dnn.readNetFromDarknet("./yolo_model/yolov4.cfg","./yolo_model/yolov4.weights")
        elif self.image_filename == "image" and self.video_filename != "video":
            network = cv2.dnn.readNetFromDarknet("./yolo_model/yolov4-tiny.cfg", "./yolo_model/yolov4-tiny.weights")

        layers_names_all = network.getLayerNames()
        layers_names_output = \
            [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        probability_minimum = 0.5
        threshold = 0.3
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

    
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, threshold)
        obs = []
        position = []
        if len(results) > 0:

            
            # Going through indexes of results
            for i in results.flatten():
            
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            
                colour_box_current = colours[class_numbers[i]].tolist()
                frame_h, frame_w = frame.shape[:2]

                x_center = int(x_min + (box_width/2))
                side = None
                if x_center > ub:
                    side = "Right"
                    position.append(side)
                elif x_center < lb:
                    side = "Left"
                    position.append(side)
                elif x_center>=lb and x_center<=ub:
                    side = "Middle"
                    position.append(side)

                

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                class_name = labels[int(class_numbers[i])]
                obs.append(class_name)
                
                
                text_box_current = '{}: {:.4f} {}'.format(class_name,
                                                    confidences[i], side)

                # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2) 
            
        return frame, obs, position
#<----- getOutputFrame end ----->


    def gen_frames(self):
        camera = cv2.VideoCapture(self.video_file_path)
        while True:
            success, frame = camera.read()
            frame= cv2.flip(frame, 1)
            # frame = cv2.resize(frame, (416, 416))
            if not success:
                pass
            else:
                out_frame, _, _ = self.getOutputFrame(frame=frame)
                ret, buffer = cv2.imencode(".jpg", out_frame)
                out_frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + out_frame + b"\r\n"
                )  # concat frame one by one and show result

    
#<----- Get Object Count ----->   
    def get_object_count(self,object_ls, position_ls):
        obj_cnt_dict = {}
        for i in list(set(object_ls)):
            obj_cnt_dict[i] = object_ls.count(i)

        
        
        obj_pos_cnt_dict = {}
        right = []
        left = []
        middle = []
        for c,p in zip(object_ls, position_ls):
            if p == "Right":
                right.append(c)
            elif p == "Left":
                left.append(c)
            elif p == "Middle":
                middle.append(c)
                

        right_dict = {}
        left_dict = {}
        middle_dict = {}
        for dict,ls in zip([right_dict, left_dict, middle_dict],[right, left, middle]):
            for i in list(set(ls)):
                dict[i] = ls.count(i)
        
        for pos, dict in zip(["Right", "Left", "Middle"], [right_dict, left_dict, middle_dict]):
            obj_pos_cnt_dict[pos] = dict
            
        print(obj_pos_cnt_dict)
        return obj_cnt_dict, obj_pos_cnt_dict

    def obj_cnt(self, obj_cnt_dict):
        text = ""
        for i in list(obj_cnt_dict.keys()):
            if text == "":
                if obj_cnt_dict[i] == 1:
                    text = text + f"There is {obj_cnt_dict[i]} {i}"
                elif obj_cnt_dict[i] > 1:
                    text = text + f"There are {obj_cnt_dict[i]} {i}s"

            elif text != "":
                if i==list(obj_cnt_dict.keys())[-1]:
                    if obj_cnt_dict[i] == 1:
                        text = text + f" and {obj_cnt_dict[i]} {i}"
                    elif obj_cnt_dict[i] >1:
                        text = text + f" and {obj_cnt_dict[i]} {i}s"
                elif i!=list(obj_cnt_dict.keys())[-1]:
                    if obj_cnt_dict[i] == 1:
                        text = text + f", {obj_cnt_dict[i]} {i}"
                    elif obj_cnt_dict[i] > 1:
                        text = text + f", {obj_cnt_dict[i]} {i}s"
        return text
#<----- Get Object Count End ----->

#<----- Get Object Count with Position ----->    
    def obj_cnt_pos(self, obj_pos_cnt_dict):
        text = ""
        for p in list(obj_pos_cnt_dict.keys()):

            for i in list(obj_pos_cnt_dict[p].keys()):
                if text == "":
                    if obj_pos_cnt_dict[p][i] == 1:
                        text = text + f"There is {obj_pos_cnt_dict[p][i]} {i} in the {p}"
                    elif obj_pos_cnt_dict[p][i] > 1:
                        text = text + f"There are {obj_pos_cnt_dict[p][i]} {i}s in the {p}"

                elif text != "":
                    if i==list(obj_pos_cnt_dict[p].keys())[-1]:
                        if obj_pos_cnt_dict[p][i] == 1:
                            text = text + f" and {obj_pos_cnt_dict[p][i]} {i} in the {p}"
                        elif obj_pos_cnt_dict[p][i] >1:
                            text = text + f" and {obj_pos_cnt_dict[p][i]} {i}s in the {p}"
                    elif i!=list(obj_pos_cnt_dict[p].keys())[-1]:
                        if obj_pos_cnt_dict[p][i] == 1:
                            text = text + f", {obj_pos_cnt_dict[p][i]} {i} in the {p}"
                        elif obj_pos_cnt_dict[p][i] > 1:
                            text = text + f", {obj_pos_cnt_dict[p][i]} {i}s in the {p}"
        return text       
#<----- Get Object Count with Position End -----> 



    def detect(self):
        img = cv2.imread(self.image_file_path)
        frame, object_ls, position_ls = self.getOutputFrame(frame=img)
        cv2.imwrite("./static/image/output.jpg", frame)
        obj_cnt_dict, obj_pos_cnt_dict = self.get_object_count(object_ls, position_ls)
        obj_cnt_text = self.obj_cnt(obj_cnt_dict)
        print(obj_cnt_text)
        obj_pos_cnt_text = self.obj_cnt_pos(obj_pos_cnt_dict)
        print(obj_pos_cnt_text)
        return obj_cnt_dict, obj_pos_cnt_dict, obj_cnt_text, obj_pos_cnt_text, object_ls
