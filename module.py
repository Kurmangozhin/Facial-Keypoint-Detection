import cv2, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import uuid



class FaceLandmark(object):
    def __init__(self, path_model):
        self.net = cv2.dnn.readNet(path_model)
        self.input_shape = (224, 224)
        self.num_keypoints = 81
        self.mean = 119
        self.std = 24.0

    def processing(self, image_path:str):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_shape)
        image_plot = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        input_blob = cv2.dnn.blobFromImage(image, scalefactor = 1/255, size = self.input_shape, swapRB = True, crop = False)
        return input_blob, image_plot

    def __call__(self, image_path:str):
        features = {"cls":[], "x":[],"y":[]}
        input_blob, image_plot = self.processing(image_path)
        self.net.setInput(input_blob)
        out = self.net.forward()
        out = out.reshape(self.num_keypoints, -1) * self.std + self.mean
        features["cls"] = [i for i in range(len(out)+1)]
        features["x"] = out[:, 0]
        features["y"] = out[:, 1]
        for i, (x, y) in enumerate(zip(features["x"],features["y"])):
            x, y  = math.ceil(x), math.ceil(y)
            cv2.putText(image_plot, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
            cv2.circle(image_plot, (x, y), 1, (0, 255, 255), cv2.FILLED)
        image_plot = cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"example/{random.randint(1,1000)}.jpg", image_plot[:,:,::-1])
        plt.imshow(image_plot)
        plt.show()
        return features


if __name__ == '__main__':
    files = glob("dataset/*.jpg")
    image_path = random.choice(files)
    face = FaceLandmark("facelandmark.onnx")
    features = face(image_path)
    #print(features)
