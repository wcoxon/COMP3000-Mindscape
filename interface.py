
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
import time

import env



class toggle():

    def __init__(self,rect,label=None):
        self.label = label

        self.widget = Button(
            plt.axes(rect),
            label
        )
        self.set_state(False)
    
    def set_state(self, state):
        self.state= state
        if state : self.widget.color,self.widget.hovercolor = '0.9', '0.95'
        else : self.widget.color,self.widget.hovercolor = "0.6", "0.65"
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

class selection():

    def __init__(self, prompt, rect, options):

        self.selected = None

        plt.axes([0,0,1,1]).text(0.5,rect[1]+rect[3],prompt,fontsize=18,ha="center")

        self.toggles = []
        toggle_height = rect[3]/len(options)
        for i,option in enumerate(options):
            toggle_bottom = rect[1]+i*toggle_height

            new_toggle = toggle([rect[0],toggle_bottom,rect[2],toggle_height],option)
            new_toggle.widget.on_clicked(self.on_clicked(i))
            self.toggles.append(new_toggle)
        
        plt.draw()
        

    def on_clicked(self,index):
        def select(val):
            self.selected = index
            for t in self.toggles:
                t.set_state(False)
            self.toggles[index].set_state(True)
        return select

plt.rcParams['toolbar']='None'
class DataBrowser():

    def __init__(self,dsm,sampleCount):
        plt.style.use('dark_background')
        
        self.dsm = dsm
        self.dataset = dsm.dataset
        self.sampleCount = sampleCount

        self.figure, self.ax = plt.subplots()
        self.gridspec = gridspec.GridSpec(nrows=2, ncols=sampleCount, height_ratios=[2, 1])

        stats = [
            "dataset name: %s" % dsm.manifest["name"],
            "volume shape: %s" % str(dsm.image_shape),
            "features: %s" % [feature["name"] for feature in dsm.manifest["features"]]
        ]

        
        self.figure.text(1,1, "\n".join(stats), fontsize=10, ha='right', va='top')


        labels = dsm.classes
        sizes = dsm.class_distribution
        plt.subplot(self.gridspec[0,0])
        plt.pie(sizes, labels=labels)

        self.load_samples()

        imgdepth = self.samples[0]["pixel_array"].shape[0]-1
        self.layer = imgdepth//2

        self.display_samples()
        layer_slider = self.display_slider()
        reroll_button = self.display_samples_button()

        proceed_button = Button(
            plt.axes([0.8, 0.3, 0.1, 0.1]),
            color="0.1",
            label="proceed"
        )
        proceed_button.on_clicked(lambda val : plt.close())

        
        plt.get_current_fig_manager().window.state('zoomed')
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
        return layer_slider
    
    def display_samples_button(self):
        new_set_button = Button(
            plt.axes([0.8, 0.5, 0.1, 0.1]),
            color="0.1",
            label="new samples"
        )
        new_set_button.on_clicked(self.update_samples)
        return new_set_button


    def load_samples(self):
        self.samples = []

        dataIterator = self.dataset.take(self.sampleCount).as_numpy_iterator()
        start_time = time.time()
        for (_inputs, _output) in dataIterator:
            loaded_time = time.time()
            print("load time:", loaded_time-start_time)
            self.samples.append({
                "pixel_array":_inputs[0],
                "diagnosis": self.dsm.classes[_output],

                **{feature["name"] : feature["tostring"](_inputs[i+1]) for i, feature in enumerate(self.dsm.manifest["features"])}
            })
            start_time=time.time()
    
    def display_samples(self):
        self.UI = []

        for i, sample in enumerate(self.samples):
            plt.subplot(self.gridspec[1,i])
            plt.axis("off")

            label_UI = plt.title("\n".join([
                *["%s: %s" % (key, value) for key, value in sample.items() if key != "pixel_array"]
            ]),fontsize = 8)

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
                "diagnosis": self.dsm.classes[_output],
                **{feature["name"] : feature["tostring"](_inputs[i+1]) for i, feature in enumerate(self.dsm.manifest["features"])}
            }
            self.samples.append(sample)

            self.UI[i]["label"].set_text("\n".join([
                    *["%s: %s" % (key, value) for key, value in sample.items() if key != "pixel_array"]
                ]))
            self.UI[i]["image"].set_data(sample["pixel_array"][self.layer])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

def selectDataset():
    
    plt.style.use('default')
    fig = plt.figure(figsize=(4,4))

    datasets=[env.ADNI3_set,env.ADNI1_set,env.ADNI1_preprocessed_set]
    dataset_options = [ds["name"] for ds in datasets]

    list_area = [0.25,0.25,0.5,0.5]
    options_menu = selection("select a dataset", list_area, dataset_options)

    fig.canvas.draw()
    fig.canvas.flush_events()

    proceed_button = Button(
        plt.axes([0.25,0,0.5,0.1]),
        label="open dataset"
    )
    def proceed(val):
        if options_menu.selected!=None:
            plt.close()
    proceed_button.on_clicked(proceed)

    plt.show()

    return datasets[options_menu.selected]

def selectArchitecture():
    
    plt.style.use('default')
    plt.figure(figsize=(4,4))

    architecture_options = ['VGG-16', 'UNet', 'ResNet']

    options_menu = selection("select an architecture",[0.25,0.25,0.5,0.5],architecture_options)

    proceed_button = Button(
        plt.axes([0.25,0,0.5,0.1]),
        label="build model"
    )
    def proceed(val):
        if options_menu.selected!=None:
            plt.close()
    proceed_button.on_clicked(proceed)

    plt.show()

    return architecture_options[options_menu.selected]

import tensorflow as tf

class PerformanceProfiler(tf.keras.callbacks.Callback):

    def __init__(self,stats):
        self.profiler = MetricsFigure(stats)
    def on_train_batch_end(self, batch, logs=None):

        self.profiler.accuracy.append(logs["sparse_categorical_accuracy"])
        self.profiler.loss.append(logs["loss"])

        self.profiler.updateCanvas()

class MetricsFigure():
    #figure = None
    #accuracyPlot = None
    #lossPlot = None

    #loss = []
    #accuracy = []

    def __init__(self,stats):
        self.loss = []
        self.accuracy = []

        plt.style.use('dark_background')

        plt.ion()
        self.figure = plt.figure(figsize=(5,8))

        self.figure.text(0,1,"\n".join(stats),va="top")
        
        self.accuracyPlot = self.figure.add_subplot(211)
        plt.title("Accuracy")
        plt.grid()

        self.lossPlot = self.figure.add_subplot(212)
        plt.title("Loss")
        plt.grid()

        
        plt.get_current_fig_manager().window.state('zoomed')
    
    def updateCanvas(self):
        self.accuracyPlot.plot(range(len(self.accuracy)),self.accuracy, color='red')
        self.lossPlot.plot(range(len(self.loss)),self.loss, color='red')

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
