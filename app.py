from flask import Flask, render_template, Response, request, redirect, url_for, session
from object_detection import ObjectDetection

import os
from werkzeug.utils import secure_filename
from speech_to_text import STT
from threading import Thread
from speech import speak
from vqa import test_model
import time


app = Flask(__name__, template_folder="./templates")

#<---------- APP CONFIG ---------->#
app.config["IMAGE_UPLOADS"] = "./static/image/"
app.config["VIDEO_UPLOADS"] = "./static/video/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG"]
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4", "MKV"]
app.config['SECRET_KEY'] = "secret key"
#<---------- APP CONFIG END ---------->#

class Compute(Thread):           
    def __init__(self, answer):
        Thread.__init__(self)
        self.answer = answer
    
    def run(self):
        time.sleep(1)
        print(self.answer)
        speak(self.answer)

@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":

        choice = request.form.get("choice")
        if choice == "od":
            od_method = request.form.get("od-method")
            if od_method == "video":
                return redirect(url_for("od_video"))
            elif od_method == "image":
                return redirect(url_for("od_upload"))

        elif choice == "vqa":
            vqa_method = request.form.get("vqa-method")
            if vqa_method == "upload":
                return redirect(url_for("vqa_upload"))
            elif vqa_method == "capture":
                return render_template(url_for("vqa_capture"))

        return render_template("homepage.html")
    else:
        return render_template("homepage.html")

#<---------- OBJECT DETECTION ---------->#


@app.route("/object-detection-options")
def object_detection_options():

    return render_template("od-option.html")

def allowed_video(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/od-video", methods=["GET", "POST"])
def od_video():
    if request.method == "POST":
        if request.files:

            video = request.files["video"]
            # CHECK IF FILE NAME IS EMPTY
            if video.filename == "":
                print("Video must have a filename")
                return redirect(request.url)

            # CHECK IF EXTENSION IS ALLOWED
            if not allowed_video(video.filename):
                print("That Video extension is not allowed")
                return redirect(request.url)

            else:
                
                filename = secure_filename(video.filename)

            # REMOVE UPLOADED FILES
            try:
                for f in os.listdir(app.config["VIDEO_UPLOADS"]):
                    os.remove(os.path.join(app.config["VIDEO_UPLOADS"], f))
            except:
                pass
            filename = filename.replace(" ", "_")
            session['filename'] = filename
            video.save(os.path.join(app.config["VIDEO_UPLOADS"], filename))

            print("Video Saved")

            return redirect(url_for("object_detection"))
    else:
        return render_template("od_video.html")


@app.route("/video_feed")
def video_feed():
    filename = session['filename']
    od = ObjectDetection(video_filename=filename)
    return Response(od.gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/od-upload", methods=["POST", "GET"])
def od_upload():

    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            # CHECK IF FILE NAME IS EMPTY
            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)

            # CHECK IF EXTENSION IS ALLOWED
            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)

            else:
                image.filename = image.filename.replace(" ","_")
                session['filename'] = image.filename
                filename = secure_filename(image.filename)

            # REMOVE UPLOADED FILES
            try:
                for f in os.listdir(app.config["IMAGE_UPLOADS"]):
                    os.remove(os.path.join(app.config["IMAGE_UPLOADS"], f))
            except:
                pass

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            print("Image Saved")

            print(image)
            return redirect(url_for('object_detection_image'))
            

    else:
        return render_template("od-upload.html")


@app.route("/object-detection-image", methods=["POST","GET"])
def object_detection_image():
    filename = session['filename']
    od = ObjectDetection(image_filename=filename)
    obj_cnt_dict, obj_pos_cnt_dict, obj_cnt_text, obj_pos_cnt_text, object_ls = od.detect()

    found_objects = [i for i in list(obj_cnt_dict.keys())]
    ### Object list with count and position
    
    right_obj = list(obj_pos_cnt_dict["Right"].keys())
    right_cnt = list(obj_pos_cnt_dict["Right"].values())
    right = zip(right_obj, right_cnt)

    middle_obj = list(obj_pos_cnt_dict['Middle'].keys())
    middle_cnt = list(obj_pos_cnt_dict['Middle'].values())
    middle = zip(middle_obj, middle_cnt)

    left_obj = list(obj_pos_cnt_dict["Left"].keys())
    left_cnt = list(obj_pos_cnt_dict["Left"].values())
    left = zip(left_obj, left_cnt)

    ### Total Objects
    total_objects = 0
    for i in list(obj_cnt_dict.values()):
        total_objects += i
    print(total_objects)
    found_objects_numbers = [obj_cnt_dict[i] for i in list(obj_cnt_dict.keys())]

    #### Object list and count
    obj_num_list = zip(found_objects, found_objects_numbers)


    output_image = "output.jpg"
    if request.method == "POST":
        choice = request.form.get("choice")
        sst = STT()
        question, text = sst.speech_to_text()
        print(question)
        if question in ["what are the objects","what are the objects in front of me","what objects are in front of me", "what objects are present in front of me"]:
            thread_b = Compute(obj_cnt_text)
            thread_b.start()
            print(obj_cnt_text)
        elif question in ["where are the objects", "where are the objects present"]:
            thread_b = Compute(obj_pos_cnt_text)
            thread_b.start()
            print(obj_pos_cnt_text)
        elif question in ["how many objects are there"]:
            if total_objects == 1:
                thread_b = Compute(f"There 1 {total_objects} object")
                thread_b.start()
            elif total_objects > 1:
                thread_b = Compute(f"There are {total_objects} objects")
                thread_b.start()
        else:
            speak("This command is not in the list")
            print(question)


        return render_template("object_detection_image.html", filename=output_image, obj_num_list=obj_num_list, total_objects=total_objects, right=right, middle=middle, left=left)



    else:
        return render_template("object_detection_image.html", filename=output_image, obj_num_list=obj_num_list, total_objects=total_objects, right=right, middle=middle, left=left)


@app.route("/object-detection")
def object_detection():
    return render_template("object_detection.html")


@app.route("/object-detection-null")
def object_detection_null():
    return render_template("object_detection_null.html")
#<---------- OBJECT DETECTION END ---------->#


#<---------- VQA ---------->#
@app.route("/query")
def query():
    print(request.query_string)
    return "No query received", 200





def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route("/vqa-upload", methods=["POST", "GET"])
def vqa_upload():

    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            # CHECK IF FILE NAME IS EMPTY
            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)

            # CHECK IF EXTENSION IS ALLOWED
            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)

            else:
                session['filename'] = image.filename
                filename = secure_filename(image.filename)

            # REMOVE UPLOADED FILES
            try:
                for f in os.listdir(app.config["IMAGE_UPLOADS"]):
                    os.remove(os.path.join(app.config["IMGAE_UPLOADS"], f))
            except:
                pass

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            print("Image Saved")

            print(image)
            return redirect(url_for("vqa_page"))

    else:
        return render_template("vqa_upload.html")
#<----- VQA PAGE ----->


@app.route("/vqa-page", methods=["POST", "GET"])
def vqa_page():
    filename = session['filename']
    if request.method == "POST":
        stt = STT()
        choice = request.form.get("choice")
        if choice == "type":
            image = filename
            question = request.form.get("question")
            print(image)
            print(question)
            session['image'] = image
            session['question'] = question
            image_path = os.path.join("./static/image/", filename)
            question = [f"<start> {question} <end>"]
            ans = test_model(image_path, question)
            ans = ans[0]
            for i in ["<start>", "<end>"]:
                ans = ans.replace(i, "")
            ans = test_model(image_path, question)
            question = question[0]
            for i in ["<start>", "<end>"]:
                question = question.replace(i, "")
            ans = ans[0]
            for i in ["<start>", "<end>"]:
                ans = ans.replace(i, "")

            return render_template("vqa_page_answer.html", ans=ans, filename = filename, question=question)



        elif choice == "mic":           
            image = filename
            image_path = os.path.join("./static/image/", filename)
            question, text = stt.speech_to_text()
            question_input = [f"<start> {question} <end>"]
            ans = test_model(image_path, question_input)
            ans = ans[0]
            for i in ["<start>", "<end>"]:
                ans = ans.replace(i, "")
            answer = f"The Answer is {ans}"
            thread_a = Compute(answer)
            thread_a.start()
            return render_template("vqa_page_answer.html", ans=ans, filename = filename, question=question)
        
           
            
    else:
        return render_template("vqa_page.html", filename=filename)
#<---------- VQA END ---------->#


#<---------- About ---------->
@app.route("/about")
def about():
    return render_template("about.html")

#<---------- About End ---------->



if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
