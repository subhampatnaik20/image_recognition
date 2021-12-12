from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd()   #get current working directory

prediction = ImagePrediction()  #instantiate the prediction 
prediction.setModelTypeAsSqueezeNet()   #set the model type i.e SqueezeNet
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "lemon.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

    #library used - imageAI  link - https://github.com/OlafenwaMoses/ImageAI
    # https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0