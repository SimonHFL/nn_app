from flask import Flask
import api


from api import * 


labeledExamples =  [((0,0,0), 1),
					((0,0,1), 0),
					((0,1,0), 1),
					((0,1,1), 0),
					((1,0,0), 1),
					((1,0,1), 0),
					((1,1,0), 1),
					((1,1,1), 0)]

inputVector = (0,0,0)

app = Flask(__name__)

@app.route('/')
def home():
	api = Api()
	result = api.evaluate(inputVector)
	return 'prediction: ' + str(result)

@app.route('/train')
def train():
	api = Api()
	result = api.train(labeledExamples)
	return 'trained'

if __name__ == "__main__":
	app.run(host="0.0.0.0", debug=True)