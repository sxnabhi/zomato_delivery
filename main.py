
from flask import Flask, request, render_template, Response
import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD.prototxt.txt", "models/MobileNetSSD.caffemodel")

app = Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route('/')
def index1():
    return render_template('index1.html')


def detect_person(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    z = []
    for i in np.arange(0, detections.shape[2]):
        if detections[0, 0, i, 1] == 15:
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                print(i)
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # print("[INFO] {}".format(label))
                z.append(img[startY:endY, startX:endX])
                # cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 0)

                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if len(z) == 0:
        return "No person detected"
    else:
        company= "Other company"
        for i in z:
            img1 = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread('zomato/zlogo2.jpg', 0)
            # Initiate SIFT detector
            sift = cv2.SIFT_create()

            # detect and compute the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # create BFMatcher object
            bf = cv2.BFMatcher()
            ##############
            # Match descriptors.
            matches = bf.match(des1, des2)

            # sort the matches based on distance
            matches = sorted(matches, key=lambda val: val.distance)
            s = 0
            for j in range(10):
                s = s + matches[j].distance
            if (s/10) <= 180:
                company= "ZOMATO"
                break
        final ="Person detected and belong to "+company
        return final




@app.route('/', methods=["POST"])
def predict():
    image = request.files['imagefile']
    img_np = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return render_template('index1.html', prediction=detect_person(img))


@app.route('/index2')
def index2():
    return render_template('index2.html')


def generate_frames():
    while True:

        ## read the camera frame
        success, frame = camera.read()
        identity = detect_person(frame)
        cv2.putText(frame, identity, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,np.random.uniform(0, 255, size = (1, 3))[0],1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(processes=5) # when running locally remove the processes argument


