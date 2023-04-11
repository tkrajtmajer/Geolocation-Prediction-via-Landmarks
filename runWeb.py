from flask import Flask, request, render_template
import cv2
import os
import main
import base64


# Flask constructor
app = Flask(__name__, template_folder='resources/templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files["video"]
        # do something
        video_path = os.path.join("resources/videos/server/", video.filename)
        video.save(video_path)
        processed_string, out_frame = main.run(video_path)

        _, buffer = cv2.imencode('.jpg', out_frame)
        # Encode bytes in Base64 format
        out_frame_proc = base64.b64encode(buffer).decode('utf-8')

        return render_template("display_res.html", result=processed_string, resim=out_frame_proc)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()

