
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

from tensorflow.keras import losses, metrics

plt.rcParams['toolbar']='None'

import env
import preprocessing
import architectures
from preprocessing import dataset_manager


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

class App():

    def __init__(self):
        self.dataset_manager = None
        self.architecture = None

        self.state = 0

        while self.state!=-1:

            if self.state==0:
                self.state = -1
                self.selectDataset()

            elif self.state==1:
                self.state = -1
                DataBrowser(self)

            elif self.state==2:
                self.state = -1
                self.selectArchitecture()

            elif self.state==3:
                self.state = -1
                self.build_model()



    def selectDataset(self):

        plt.style.use('default')
        fig = plt.figure(figsize=(4,4))

        datasets=[env.ADNI3_set,env.ADNI1_set,env.ADNI1_preprocessed_set]
        dataset_options = [ds["name"] for ds in datasets]

        list_area = [0.25,0.25,0.5,0.5]
        options_menu = selection("select a dataset", list_area, dataset_options)


        proceed_button = Button(plt.axes([0.25,0,0.5,0.1]),label="open dataset")

        def proceed(event):
            if options_menu.selected==None: return
            self.dataset_manager = dataset_manager(datasets[options_menu.selected])
            self.state=1
            plt.close()

        proceed_button.on_clicked(proceed)

        plt.show()
    
    def selectArchitecture(self):
    
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
                self.architecture = architecture_options[options_menu.selected]
                self.state = 3
        proceed_button.on_clicked(proceed)

        plt.show()


    def build_model(self):
        model = architectures.buildModel(self.dataset_manager, self.architecture)

        epoch_samples = 1024

        boundaries = [0.25*epoch_samples/env.batchSize, 0.5*epoch_samples/env.batchSize, 0.75*epoch_samples/env.batchSize]
        values = [1e-4, 5e-5, 1e-5, 1e-6]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss = losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [metrics.SparseCategoricalAccuracy()]
        )
        model.summary()

        training_data = self.dataset_manager.dataset.take(epoch_samples)#.cache("training_cache2")

        stats = [
            "dataset: %s" % self.dataset_manager.manifest["name"],
            "architecture: %s" % self.architecture,
            "epochs: %s" % env.epochs,
            "batch size: %s" % env.batchSize
        ]

        model.fit(
            x=training_data.batch(env.batchSize),
            steps_per_epoch=epoch_samples,
            epochs=env.epochs,
            batch_size=env.batchSize,
            callbacks=[PerformanceProfiler(stats)],
            shuffle=True,
            class_weight=self.dataset_manager.class_weight
        )

        import matplotlib.pyplot as plt
        plt.show(block=True) # to prevent it from closing when training is done



class DataBrowser():

    def __init__(self,app):
        self.app = app
        self.dsm = app.dataset_manager
        self.dataset = self.dsm.dataset
        self.sampleCount = 3
        self.page = 0

        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots()
        self.gridspec = gridspec.GridSpec(nrows=2, ncols=self.sampleCount, height_ratios=[2, 1])

        stats = [
            "dataset name: %s" % self.dsm.manifest["name"],
            "volume shape: %s" % str(self.dsm.image_shape),
            "features: %s" % [feature["name"] for feature in self.dsm.manifest["features"]]
        ]

        
        self.figure.text(1,1, "\n".join(stats), fontsize=10, ha='right', va='top')


        labels = self.dsm.classes
        sizes = self.dsm.class_distribution
        plt.subplot(self.gridspec[0,0])
        plt.pie(sizes, labels=labels)

        self.load_samples()

        imgdepth = self.samples[0]["pixel_array"].shape[0]-1
        self.layer = imgdepth//2

        self.display_samples()
        self.layer_slider = self.display_slider()
        self.next_button, self.prev_button = self.page_buttons()

        self.return_button = Button(
            plt.axes([0.0, 0.5, 0.05, 0.05]),
            color="0.1",
            label="back"
        )
        self.return_button.on_clicked(lambda val : self.return_to_datasets())

        self.proceed_button = Button(
            plt.axes([0.9, 0.5, 0.05, 0.05]),
            color="0.1",
            label="proceed"
        )
        self.proceed_button.on_clicked(lambda val : self.proceed_to_model())

        
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
    
    def page_buttons(self):
        next_button = Button(
            plt.axes([0.9, 0.3, 0.05, 0.05]),
            color="0.1",
            label="next row"
        )
        prev_button = Button(
            plt.axes([0.0, 0.3, 0.05, 0.05]),
            color="0.1",
            label="prev row"
        )

        next_button.on_clicked(lambda label : self.update_samples(self.page+1))
        prev_button.on_clicked(lambda label : self.update_samples(self.page-1))

        return next_button,prev_button

    def load_samples(self):
        self.samples = []
        #generate set of image paths
        paths_set = tf.data.Dataset.list_files(self.dsm.manifest["images_path"],shuffle=False).skip(self.page*self.sampleCount).take(self.sampleCount)

        for path in paths_set:
            (_inputs, _output) = self.dsm.generate_sample(path)

            self.samples.append({
                "pixel_array":_inputs[0],
                "diagnosis": self.dsm.classes[_output],
                **{feature["name"] : feature["tostring"](_inputs[i+1]) for i, feature in enumerate(self.dsm.manifest["features"])}
            })

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

    def update_samples(self,page):
        self.samples = []
        self.page = page
        paths_set = tf.data.Dataset.list_files(self.dsm.manifest["images_path"],shuffle=False).skip(self.page*self.sampleCount).take(self.sampleCount)
        
        for i, path in enumerate(paths_set):

            (_inputs, _output) = self.dsm.generate_sample(path)

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

    def update_layer(self, layer):
        self.layer = layer
        for i in range(self.sampleCount):
                self.UI[i]["image"].set_data(self.samples[i]["pixel_array"][layer])

    def proceed_to_model(self):
        plt.close()
        self.app.state = 2
        #selectArchitecture()
    
    def return_to_datasets(self):
        plt.close()
        self.app.state = 0




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
