import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons



architecture = 'VGG-16'


class DataBrowser():
    figure = None
    dataset = None
    sampleCount = 0
    samples = []
    UI = []
    layer = 0

    def __init__(self,dataset,sampleCount):
        self.dataset = dataset
        self.sampleCount = sampleCount
        self.figure = plt.figure(figsize=(10,4))

        self.layer = 0

        self.load_samples()

        self.display_samples()

        layer_slider = Slider(
            ax=plt.axes([0.1,0.1,0.8,0.05]), 
            label="layer",
            valmin=0, 
            valmax=self.samples[0]["pixel_array"].shape[0]-1, 
            valinit=0, 
            valstep=1
        )
        layer_slider.on_changed(self.update_layer)


        new_set_button = Button(
            plt.axes([0.9, 0.5, 0.1, 0.1]),
            label="new selection"
        )
        new_set_button.on_clicked(self.update_samples)

        plt.show()
    
    def load_samples(self):
        self.samples = []

        dataIterator = self.dataset.take(self.sampleCount).as_numpy_iterator()
        for (_inputs, _output) in dataIterator:
            self.samples.append({
                "pixel_array":_inputs[0],
                "age": str(_inputs[1]),
                "sex": ["Male","Female"][_inputs[2]],
                "diagnosis": ['CN','SMC','EMCI','MCI','LMCI','AD'][_output]
            })
    
    def display_samples(self):
        self.UI = []

        for i, sample in enumerate(self.samples):
            plt.subplot(1,self.sampleCount,1+i)
            plt.axis("off")

            label_UI = plt.title("\n".join([
                "DX: " + sample["diagnosis"],
                "Age: " + sample["age"],
                "Sex: " + sample["sex"]
            ]))

            slice_UI = plt.imshow(sample["pixel_array"][0],cmap='gray',vmin=0, vmax=1)

            self.UI.append({
                "label":label_UI, 
                "image":slice_UI
            })
    
    def update_layer(self, layer):
        self.layer = layer
        for i in range(self.sampleCount):
                self.UI[i]["image"].set_data(self.samples[i]["pixel_array"][layer])

    def update_samples(self,val=None):

        #self.load_samples()
        self.samples = []
        dataIterator = self.dataset.take(self.sampleCount).as_numpy_iterator()
        for i, (_inputs, _output) in enumerate(dataIterator):
            
            sample = {
                "pixel_array":_inputs[0],
                "age": str(_inputs[1]),
                "sex": ["Male","Female"][_inputs[2]],
                "diagnosis": ['CN','SMC','EMCI','MCI','LMCI','AD'][_output]
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


        #for i, sample in enumerate(self.samples):
        #        self.UI[i]["label"].set_text("\n".join([
        #            "DX: " + sample["diagnosis"],
        #            "Age: " + sample["age"],
        #            "Sex: " + sample["sex"]
        #        ]))
        #        self.UI[i]["image"].set_data(sample["pixel_array"][0])



def selectArchitecture():
    plt.figure(figsize=(4,4))

    radio = RadioButtons(
        plt.axes([0,0,1,1]), 
        ('VGG-16', 'UNet', 'ResNet')
    )
    def setArchitecture(label):
        global architecture
        architecture = label
    radio.on_clicked(setArchitecture)

    plt.show()

    return architecture


class MetricsFigure():
    figure = None
    accuracyPlot = None
    lossPlot = None

    def __init__(self):
        plt.ion()
        self.figure = plt.figure()

        self.accuracyPlot = self.figure.add_subplot(211)
        plt.grid()

        self.lossPlot = self.figure.add_subplot(212)
        plt.grid()
    
    def plotAccuracy(self, accuracy):
        self.accuracyPlot.plot(range(len(accuracy)),accuracy)

    def plotLoss(self, loss):
        self.lossPlot.plot(range(len(loss)),loss)
    
    def updateCanvas(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
