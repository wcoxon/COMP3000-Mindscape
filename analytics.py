
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

import tensorflow as tf

import time
import json
import env

plt.rcParams['toolbar']='None'

class PerformanceProfiler(tf.keras.callbacks.Callback):
    def __init__(self,app):
        self.app = app
        self.start_time = time.time()

        stats = [
            "dataset: %s" % app.dataset_manager.manifest["name"],
            "features: %s" % [x["name"] for x in app.dataset_manager.manifest["features"]],
            "architecture: %s" % app.architecture,
            "epochs: %s" % env.epochs,
            "batch size: %s" % env.batchSize
        ]

        self.profiler = MetricsFigure(stats)

    def on_train_batch_end(self, batch, logs=None):

        self.profiler.accuracy.append(logs["sparse_categorical_accuracy"])
        self.profiler.loss.append(logs["loss"])

        self.profiler.updateCanvas()
    
    def on_train_end(self, logs=None):
        training_duration = time.time()-self.start_time


        report_name = "report_%s"%time.strftime("%d-%m-%Y_%H-%M-%S")

        report_data = {
            "header":{
                "name":report_name,
                "dataset" : self.app.dataset_manager.manifest["name"],
                "features" : [x["name"] for x in self.app.dataset_manager.manifest["features"]],
                "architecture" : self.app.architecture,
                "batch size" : env.batchSize,
                "samples per epoch" : env.epochSize,
                "epochs":env.epochs,
                "training duration":training_duration
            },
            "loss":self.profiler.loss,
            "accuracy":self.profiler.accuracy
        }
        with open("reports/%s.json"%report_name,"w") as file:
            json.dump(report_data,file)
        self.app.reports.append(report_data)
        self.app.state=env.report_state
        plt.close()

        return super().on_train_end(logs)

class MetricsFigure():
    def __init__(self,stats):
        self.loss = []
        self.accuracy = []

        #plt.style.use('dark_background')
        plt.ion()

        self.figure = plt.figure(figsize=(5,8))

        self.figure.text(0,1,"\n".join(stats),va="top")
        
        self.accuracyPlot = self.figure.add_subplot(211)
        plt.title("Accuracy / Batch")
        plt.grid(True)

        self.lossPlot = self.figure.add_subplot(212)
        plt.title("Loss / Batch")
        plt.grid(True)
        
        plt.get_current_fig_manager().window.state('zoomed')
    
    def updateCanvas(self):
        self.accuracyPlot.plot(range(len(self.accuracy)),self.accuracy, color='red')
        self.lossPlot.plot(range(len(self.loss)),self.loss, color='red')

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
