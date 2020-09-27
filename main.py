from flask import Flask, request, render_template
from pred import predictResp, cracks_wheezles
import base64
from show_image import plot_spectogram


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home() :
    if request.method == 'POST' :
        audio = request.files['file']
        if audio.filename == '' :
            return "not uploaded"
        else :
            filename = "Audio/" + audio.filename
            audio.save(filename)
            ans = predictResp(filename)
        return ans
    return render_template('home.html')

@app.route('/show_image',methods=['GET','POST'])
def show_image() :
    if request.method == 'POST' :
        audio = request.files['file']
        filename = "Audio/" + audio.filename

        print("heheheheheeheh",filename)

        plot_spectogram(filename)
    
        with open("Spectograms/spectogram_image.png", "rb") as image_file:
            encoded_strings = base64.b64encode(image_file.read())
    
        return encoded_strings
    else :
        return "no file"
        

@app.route('/wc',methods=['GET','POST'])
def wc() :
    return cracks_wheezles()



app.run(host='192.168.1.107')