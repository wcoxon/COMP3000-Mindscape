import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.widgets import Slider, Button, RadioButtons
import time

import csv

import env
from env import classes, num_classes, label_map

import numpy as np

from preprocessing import class_distribution


class DataBrowser():
    figure = None
    dataset = None
    sampleCount = 0
    samples = []
    UI = []
    layer = 0
    gridspec = None

    def __init__(self,dataset,sampleCount,class_dist):
        plt.style.use('dark_background')
        self.dataset = dataset
        self.sampleCount = sampleCount

        self.figure = plt.figure(figsize=(10,5))
        self.gridspec = gridspec.GridSpec(nrows=2, ncols=sampleCount, height_ratios=[2, 1])

        labels = classes
        sizes = class_dist

        plt.subplot(self.gridspec[0,0])
        plt.pie(sizes, labels=labels)

        self.load_samples()

        imgdepth = self.samples[0]["pixel_array"].shape[0]-1
        self.layer = imgdepth//2

        self.display_samples()
        self.display_slider()

        self.display_samples_button()

        plt.show()

    def display_slider(self):
        volume_depth = self.samples[0]["pixel_array"].shape[0]-1

        layer_slider = Slider(
            ax=plt.axes([0.1,0.1,0.8,0.025]), 
            label="layer",
            valmin=0, 
            valmax=volume_depth, 
            valinit=self.layer, 
            valstep=1
        )
        layer_slider.on_changed(self.update_layer)
    
    def display_samples_button(self):
        new_set_button = Button(
            plt.axes([0.9, 0.5, 0.1, 0.1]),
            color="0.1",
            label="new samples"
        )
        new_set_button.on_clicked(self.update_samples)


    def load_samples(self):
        self.samples = []

        dataIterator = self.dataset.take(self.sampleCount).as_numpy_iterator()
        start_time = time.time()
        for (_inputs, _output) in dataIterator:
            loaded_time = time.time()
            print("load time:", loaded_time-start_time)
            self.samples.append({
                "pixel_array":_inputs[0],
                "age": str(_inputs[1]),
                "sex": ["Male","Female"][_inputs[2][1]],
                "diagnosis": classes[_output]
            })
            start_time=time.time()
    
    def display_samples(self):
        self.UI = []

        for i, sample in enumerate(self.samples):
            plt.subplot(self.gridspec[1,i])
            plt.axis("off")

            label_UI = plt.title("\n".join([
                "DX: " + sample["diagnosis"],
                "Age: " + sample["age"],
                "Sex: " + sample["sex"]
            ]),fontsize = 12)

            slice_UI = plt.imshow(sample["pixel_array"][self.layer],cmap='gray',vmin=0, vmax=1)

            self.UI.append({
                "label":label_UI, 
                "image":slice_UI
            })
    
    def update_layer(self, layer):
        self.layer = layer
        for i in range(self.sampleCount):
                self.UI[i]["image"].set_data(self.samples[i]["pixel_array"][layer])

    def update_samples(self,val=None):

        self.samples = []
        dataIterator = self.dataset.take(self.sampleCount).as_numpy_iterator()
        for i, (_inputs, _output) in enumerate(dataIterator):
            
            sample = {
                "pixel_array":_inputs[0],
                "age": str(_inputs[1]),
                "sex": ["Male","Female"][_inputs[2][1]],
                "diagnosis": classes[_output]
            }

            self.samples.append(sample)

            self.UI[i]["label"].set_text("\n".join([
                    "DX: " + sample["diagnosis"],
                    "Age: " + sample["age"],
                    "Sex: " + sample["sex"]
                ]))
            self.UI[i]["image"].set_data(sample["pixel_array"][self.layer])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()



def selectArchitecture():
    
    plt.style.use('default')
    plt.figure(figsize=(4,4))

    radio = RadioButtons(
        plt.axes([0,0,1,1]), 
        ('VGG-16', 'UNet', 'ResNet')
    )
    def setArchitecture(label):
        #global architecture
        env.architecture = label
    radio.on_clicked(setArchitecture)

    plt.show()

    return env.architecture

import tensorflow as tf

class PerformanceProfiler(tf.keras.callbacks.Callback):

    def __init__(self):
        self.profiler = MetricsFigure()
    def on_train_batch_end(self, batch, logs=None):

        self.profiler.accuracy.append(logs["sparse_categorical_accuracy"])
        #profiler.plotAccuracy(accuracy)

        self.profiler.loss.append(logs["loss"])
        #profiler.plotLoss(loss)

        self.profiler.updateCanvas()

class MetricsFigure():
    figure = None
    accuracyPlot = None
    lossPlot = None

    loss = []
    accuracy = []

    def __init__(self):
        self.loss = []
        self.accuracy = []

        plt.ion()
        self.figure = plt.figure(figsize=(5,8))

        stats = [
            "epochs: %s" % env.epochs,
            "architecture: %s" % env.architecture,
            "batch size: %s" % env.batchSize,
            "dataset: %s" % env.dataset_props["name"]
        ]

        self.figure.text(0,0,"\n".join(stats))
        
        self.accuracyPlot = self.figure.add_subplot(211)
        plt.title("Accuracy")
        plt.grid()

        self.lossPlot = self.figure.add_subplot(212)
        plt.title("Loss")
        plt.grid()
    
    def updateCanvas(self):
        self.accuracyPlot.plot(range(len(self.accuracy)),self.accuracy, color='red')
        self.lossPlot.plot(range(len(self.loss)),self.loss, color='red')

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
