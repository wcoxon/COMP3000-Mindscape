
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

from tkinter import filedialog

import tensorflow as tf
from tensorflow.keras import losses, metrics

import json

import env
import preprocessing
import architectures
from DataBrowser import DataBrowser
from analytics import PerformanceProfiler

plt.rcParams['toolbar']='None'
plt.ion()

class toggle():
    def __init__(self,rect,label=None):
        self.label = label
        self.widget = Button(plt.axes(rect),label)
        self.set_state(False)
    
    def set_state(self, state):
        self.state = state

        if state : self.widget.color,self.widget.hovercolor = '0.9', '0.95'
        else : self.widget.color,self.widget.hovercolor = "0.6", "0.65"
        
        self.widget.ax.set_facecolor(self.widget.color)

class selection():
    def __init__(self, prompt, rect, options,multi=False):
        self.toggles = []
        self.selected = []
        self.multi=multi
        self._click = self.multi_select if multi else self.select
        if len(options)==0: return

        label_margin = 0.05
        plt.gcf().text(rect[0]+0.5*rect[2], rect[1]+rect[3]+label_margin, prompt, fontsize=18, ha="center")

        button_height = rect[3]/len(options)
        for i,option in enumerate(options):
            button_bottom = (rect[1]+rect[3])-(i+1)*button_height
            new_toggle = toggle([rect[0],button_bottom,rect[2],button_height],option)
            new_toggle.widget.on_clicked((lambda i: lambda event: self._click(i))(i))
            self.toggles.append(new_toggle)
        
    def on_clicked(self,func):
        self._click = func
        
    def multi_select(self,index):
        self.toggles[index].set_state(not self.toggles[index].state)
        self.selected = [i for i, t in enumerate(self.toggles) if t.state]

    def select(self,index):
        self.selected = [index]
        for i,t in enumerate(self.toggles):
            t.set_state(i==index)



class App():

    def __init__(self):
        self.reports = []

        self.dataset_manager = None
        self.architecture = None

        self.state = 0

        page_map={
            -1 : lambda : None,
            env.report_state : self.displayReports,
            env.data_select_state : self.selectDataset,
            env.data_browse_state : lambda : DataBrowser(self),
            env.architecture_select_state : self.selectArchitecture,
            env.train_state : self.build_model,
            env.feature_select_state : self.selectFeatures
        }

        while self.state!=-1:
            page_func = page_map[self.state]
            self.state=-1
            page_func()
    
    def displayReports(self):

        colours = ['red','green','blue','orange','cyan','purple','brown','pink','yellow']

        fig = plt.figure(figsize=(5,8))

        def load_report(event):
            filename = filedialog.askopenfilename(
                initialdir = "./reports/",
                title = "Select a File",
                filetypes = (("JSON files","*.json"),)
            )
            with open(filename) as f:
                data = json.load(f)
            self.reports.append(data)

            self.state=env.report_state
            plt.close()
        
        def new_report(event):
            self.state=env.data_select_state
            plt.close()


        report_info = fig.text(0,1,"",va="top")

        report_names = [report["header"]["name"] for report in self.reports]
        reports_selection = selection("reports:", [0,0,0.1,0.5], report_names)

        def select_report(index):
            print(index)
            reports_selection.select(index)
            report_info.set_text("\n".join(["%s: %s" % (k,v) for k,v in self.reports[index]["header"].items()]))
            report_info.set_color(colours[index])
            fig.canvas.draw()
            fig.canvas.flush_events()
        reports_selection.on_clicked(select_report)

        def close_report(event):
            if reports_selection.selected[0]==None: return
            self.reports.remove(self.reports[reports_selection.selected[0]])
            self.state=env.report_state
            plt.close()


        load_button = Button(plt.axes([0.2,0.0,0.2,0.05]),label="load report")
        new_button = Button(plt.axes([0.4,0.0,0.2,0.05]),label="create new report")
        close_button = Button(plt.axes([0.6,0.0,0.2,0.05]),label="close selected report")

        load_button.on_clicked(load_report)
        new_button.on_clicked(new_report)
        close_button.on_clicked(close_report)

        accuracyPlot = fig.add_subplot(211)
        plt.title("Accuracy / Batch")
        plt.grid(True)

        lossPlot = fig.add_subplot(212)
        plt.title("Loss / Batch")
        plt.grid(True)
        
        for i,report in enumerate(self.reports):
            accuracyPlot.plot(range(len(report["accuracy"])),report["accuracy"], color=colours[i])
            lossPlot.plot(range(len(report["loss"])),report["loss"], color=colours[i],label=report_names[i])
            plt.legend()
        
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show(block=True)

    def selectDataset(self):

        fig = plt.figure(figsize=(4,4))

        datasets=[env.ADNI3_set,env.ADNI1_set,env.ADNI1_preprocessed_set]
        dataset_options = [ds["name"] for ds in datasets]

        list_area = [0.25,0.25,0.5,0.5]
        options_menu = selection("select a dataset", list_area, dataset_options)

        def proceed(event):
            if options_menu.selected==None: return
            self.dataset_manager = preprocessing.dataset_manager(datasets[options_menu.selected[0]])
            self.state=env.feature_select_state
            plt.close()

        def back(event):
            self.state = env.report_state
            plt.close()

        proceed_button = Button(plt.axes([0.25,0,0.5,0.1]),label="open dataset")
        back_button = Button(plt.axes([0.0,0,0.25,0.1]),label="back")

        proceed_button.on_clicked(proceed)
        back_button.on_clicked(back)
        plt.show(block=True)


    def selectFeatures(self):
        feature_options = [feature_name for feature_name in env.features.keys()]
        
        fig = plt.figure(figsize=(4,4))


        list_area = [0.25,0.25,0.5,0.5]
        options_menu = selection("select features", list_area, feature_options,True)


        def proceed(event):
            if options_menu.selected==None: return
            self.dataset_manager.manifest["features"] = [env.features[feature_options[i]] for i in options_menu.selected]
            self.dataset_manager.dataset = self.dataset_manager.generator_dataset()
            self.state=env.data_browse_state
            plt.close()

        def back(event):
            self.state = env.data_select_state
            plt.close()

        proceed_button = Button(plt.axes([0.25,0,0.5,0.1]),label="open dataset")
        back_button = Button(plt.axes([0.0,0,0.25,0.1]),label="back")

        proceed_button.on_clicked(proceed)
        back_button.on_clicked(back)

        plt.show(block=True)

    def selectArchitecture(self):
    
        plt.figure(figsize=(8,4))

        architecture_options = ['VGG-16', 'UNet', 'ResNet']
        batch_options = [1,2,4,8]
        epoch_size_options = [64,128,256,512,1024]
        epoch_count_options = [1,2,4,8,16]

        architecture_menu = selection("architecture",[0.0,0.25,0.25,0.5],architecture_options)
        batch_menu = selection("batch size",[0.25,0.25,0.25,0.5],batch_options)
        epoch_size_menu = selection("samples / epoch",[0.5,0.25,0.25,0.5],epoch_size_options)
        epoch_count_menu = selection("epochs",[0.75,0.25,0.25,0.5],epoch_count_options)

        proceed_button = Button(plt.axes([0.25,0,0.5,0.1]),label="build model")
        return_button = Button(plt.axes([0.0,0,0.25,0.1]),label="back")

        def proceed(event):
            if any([selector.selected[0]==None for selector in [architecture_menu,batch_menu,epoch_size_menu,epoch_count_menu]]): return
            self.architecture = architecture_options[architecture_menu.selected[0]]
            env.batchSize = batch_options[batch_menu.selected[0]]
            env.epochSize = epoch_size_options[epoch_size_menu.selected[0]]
            env.epochs = epoch_count_options[epoch_count_menu.selected[0]]
            self.state = env.train_state
            plt.close()
        def back(event):
            self.state = env.data_browse_state
            plt.close()

        proceed_button.on_clicked(proceed)
        return_button.on_clicked(back)
        plt.show(block=True)

    def build_model(self):
        model = architectures.buildModel(self.dataset_manager, self.architecture)

        boundaries = [0.25*env.epochSize/env.batchSize, 0.5*env.epochSize/env.batchSize, 0.75*env.epochSize/env.batchSize]
        values = [1e-4, 5e-5, 1e-5, 1e-6]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss = losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [metrics.SparseCategoricalAccuracy()]
        )
        model.summary()

        training_data = self.dataset_manager.dataset.take(env.epochSize)#.cache("training_cache2")

        model.fit(
            x=training_data.batch(env.batchSize),
            epochs=env.epochs,
            batch_size=env.batchSize,
            callbacks=[PerformanceProfiler(self)],
            shuffle=True,
            class_weight=self.dataset_manager.class_weight
        )

        plt.show(block=True)
