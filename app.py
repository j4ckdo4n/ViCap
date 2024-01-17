import os
import time

from flask import Flask, jsonify, render_template, request
#from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from models.clipcap import Predictor_ClipCap
from models.smallcap import Predictor_SmallCap
from logging.config import dictConfig
dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
			"file": {
                "class": "logging.FileHandler",
                "filename": "system.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console","file"]},
    }
)
app = Flask(__name__, template_folder='./')
#run_with_ngrok(app)

UPLOAD_FOLDER = './static/src/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def show_template():
	return render_template("./static/main.html")


@app.route("/extract", methods=[ 'POST'])
def extract():
	if request.method == 'POST':
		# Get image from POST request
		f = request.files['file']
		file_name = secure_filename(f.filename)
		
		# Save image to ./uploads
		f.save(os.path.join(
			app.config['UPLOAD_FOLDER'], file_name))

		start = time.time()

		file_full_path = "./static/src/uploads/" + file_name
		dataset_name = request.form['method']
		#print(file_full_path)
		caption2,retrieval_caps = predictor2.predict(file_full_path,dataset_name)
		cap1, cap2, cap3, cap4 = retrieval_caps
		

		app.logger.info("caption SmallCap predict success")
		caption1 = predictor1.predict(file_full_path,dataset_name)
		app.logger.info("caption ClipCap predict success")
		

		


		

		ret = {
			"status": "OK",  # any status <> "OK" means failed to extract
			"dataset": dataset_name,
			"caption_clipcap": caption1,
			"caption_smallcap": caption2,
			"elapsed_time": time.time() - start,
			"file": file_name,
			"ret1": cap1,
			"ret2": cap2,
			"ret3": cap3,
			"ret4": cap4,
		}
		#logging respone
		app.logger.info("response: %s", ret)
		#print(ret)
		return jsonify(ret)

predictor1=Predictor_ClipCap()
predictor2=Predictor_SmallCap()
if __name__ == '__main__':
	
	# predictor.load_detect_model()
	# predictor.load_reg_model()
	# predictor.load_craft_model()
	#run_with_ngrok(app)
	# print('ngrok')
	app.run(host="0.0.0.0",port=5000)
	#app.run()