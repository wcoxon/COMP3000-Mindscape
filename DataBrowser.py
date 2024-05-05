
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

import tensorflow as tf
import env


plt.rcParams['toolbar']='None'

class DataBrowser():
    def __init__(self,app):
        self.app = app
        self.dsm = app.dataset_manager
        self.dataset = self.dsm.dataset
        self.sampleCount = 3
        self.page = 0

        #plt.style.use('dark_background')
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

        self.layer = self.dsm.image_shape[0]//2

        self.display_samples()
        self.layer_slider = self.display_slider()
        self.prev_button, self.next_button = self.page_buttons()

        self.return_button = Button(plt.axes([0.0, 0.5, 0.05, 0.05]),label="return")
        self.return_button.on_clicked(self.return_to_datasets)

        self.proceed_button = Button(plt.axes([0.9, 0.5, 0.05, 0.05]),label="setup model")
        self.proceed_button.on_clicked(self.proceed_to_model)

        plt.get_current_fig_manager().window.state('zoomed')
        plt.show(block=True)

    def display_slider(self):
        volume_depth = self.dsm.image_shape[0]

        layer_slider = Slider(
            ax=plt.axes([0.9,0.15,0.025,0.2]), 
            label="layer",
            valmin=0, 
            valmax=volume_depth-1, 
            valinit=self.layer, 
            valstep=1,
            orientation="vertical"
        )
        layer_slider.on_changed(self.update_layer)
        return layer_slider
    
    def page_buttons(self):
        prev_button = Button(plt.axes([0.4, 0.0, 0.1, 0.05]),label="prev row")
        prev_button.on_clicked(lambda label : self.update_samples(self.page-1))

        next_button = Button(plt.axes([0.5, 0.0, 0.1, 0.05]),label="next row")
        next_button.on_clicked(lambda label : self.update_samples(self.page+1))

        return prev_button, next_button

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

            self.UI[i]["label"].set_text("\n".join(["%s: %s" % (key, value) for key, value in sample.items() if key != "pixel_array"]))
            self.UI[i]["image"].set_data(sample["pixel_array"][self.layer])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def update_layer(self, layer):
        self.layer = layer
        for i in range(self.sampleCount):
                self.UI[i]["image"].set_data(self.samples[i]["pixel_array"][layer])

    def proceed_to_model(self,event):
        plt.close()
        self.app.state = env.architecture_select_state
    
    def return_to_datasets(self,event):
        plt.close()
        self.app.state = env.report_state