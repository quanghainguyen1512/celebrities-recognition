from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename


from utils import *
from detector import detect_faces
from PIL import Image
from visualization import show_results
from align import *

from keras import backend as K 
from face_detect import face_detect

#https://www.youtube.com/watch?v=JJSoEo8JSnc


UPLOAD_FOLDER = './face_ss'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = face_detect()

@app.route("/")
def index():
	return render_template('index.html')


@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "nani?"
        else:
            f = request.files['file']
            type = secure_filename(f.filename).split('.')[1]
            if type not in ALLOWED_EXTENSIONS:
                return 'Invalid type of file'
            if f :
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) )
                f.save(filename)

                fn = secure_filename(f.filename)[:-4]

                # resize the image before face detection
                min_side = 512
                img = cv2.imread(filename)
                size = img.shape
                h, w  = size[0], size[1]
                if max(w, h) > min_side:
                    img_pad = process_image(img)
                else:
                    img_pad = img
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_resize.png' ), img_pad)
                
                # face detection        
                img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_resize.png' )) 
                bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
                pic_face_detect = show_results(img, bounding_boxes, landmarks) # visualize the results
                pic_face_detect.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_landmark.png' ) )
                crop_size = 224
                scale = crop_size / 112
                reference = get_reference_facial_points(default_square = True) * scale
                facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)   
                pic_face_crop = img_warped.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_crop.png' ) )

                # face recognition    
                cleb_name = model.predict(os.path.join(app.config['UPLOAD_FOLDER'], f'{fn}_crop.png'))
                return render_template('done.html', 
                    original_image=f'http://celeb.kyanon.digital/predictions/{fn}_resize.png',
                    user_image = f'http://celeb.kyanon.digital/predictions/{fn}_landmark.png',
                    crop_image = f'http://celeb.kyanon.digital/predictions/{fn}_crop.png',
                    name=cleb_name
                )
    	        # return render_template("done.html",
                #     original_image = f'http://celeb.kyanon.digital/predictions/{f.filename}_resize.png',  
                #     user_image = f'http://celeb.kyanon.digital/predictions/{f.filename}_landmark.png' ,
                #     crop_image = f'http://celeb.kyanon.digital/predictions/{f.filename}_crop.png',
                #     name =  cleb_name)
 		

if __name__ == '__main__':
    #remove debug=True before actual deployment smh
   app.run(debug = True)
