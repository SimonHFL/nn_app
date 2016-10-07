from neural_network import *
from api import * 
import pickle 

labeledExamples =  [((0,0,0), 1),
					((0,0,1), 0),
					((0,1,0), 1),
					((0,1,1), 0),
					((1,0,0), 1),
					((1,0,1), 0),
					((1,1,0), 1),
					((1,1,1), 0)]

api = Api()
#api.create()
api.train(labeledExamples)
