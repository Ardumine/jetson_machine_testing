print("Staring the program....")
print("1/15")
from jetcam.usb_camera import USBCamera

import torchvision.transforms as transforms

from dataset import ImageClassificationDataset

import base64
print("5/15")

import ipywidgets
print("6/15")

import traitlets
print("7/15")

from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from IPython.display import display


from jetcam.utils import bgr8_to_jpeg


import torch
import torchvision

import threading
import os
import time
print("13/15")

from utils import preprocess
print("14/15")
import torch.nn.functional as F

print("LIB INI DONE")



###################CONFIG

load_model_from_file = False # if activated, the program will load the model from the selected file in the model_to_load. If not activated, the program will learn from source.
#############LEARN FROM SCRATCH ON START CONFIG

TASK = 'cans_3' #Here put your folder name in classification, without the _A
epochs_to_train = 3


#############LOAD MODEL ON START CONFIG
CATEGORIES = [ '7_up_broken', '7_up_ok', 'icetea_broken', 'icetea_ok'] #our catagories, that our ai will identify 

model_to_load = "/nvdli-nano/data/classification/my_model5.pth" # here put the path of your model. Make sure its of the docker conatiner.






camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number
current_imagee = None

camera.running = True

print("Camera started!")



DATASETS = ['A'] #Here put yoor datasets id's

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not load_model_from_file:
    CATEGORIES = []
    folders = os.listdir('/nvdli-nano/data/classification/' + TASK + '_' + DATASETS[0])
    for folder in folders:
        CATEGORIES.append(folder)
                     
datasets = {}
for name in DATASETS:
    
    datasets[name] = ImageClassificationDataset('/nvdli-nano/data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)
    
print("{} task with {} categories defined".format(TASK, CATEGORIES))

print("Please wait... Starting program of AI.")


# initialize active dataset
dataset = datasets[DATASETS[0]]

# unobserve all callbacks from camera in case we are running this cell for second time
camera.unobserve_all()




# create image preview
camera_widget = ipywidgets.Image()
#traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

# create widgets
dataset_widget = ipywidgets.Dropdown(options=DATASETS, description='dataset')
category_widget = ipywidgets.Dropdown(options=dataset.categories, description='category')
count_widget = ipywidgets.IntText(description='count')

# manually update counts at initialization
count_widget.value = dataset.get_count(category_widget.value)

# sets the active dataset
def set_dataset(change):
    global dataset
    dataset = datasets[change['new']]
    count_widget.value = dataset.get_count(category_widget.value)
dataset_widget.observe(set_dataset, names='value')

# update counts when we select a new category
def update_counts(change):
    count_widget.value = dataset.get_count(change['new'])
category_widget.observe(update_counts, names='value')

# save image for category and update counts
def save(c):
    dataset.save_entry(camera.value, category_widget.value)
    #dataset.save_entry2(camera.value, category_widget.value,  dataset.get_count(category_widget.value) )
    count_widget.value = dataset.get_count(category_widget.value)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget]), dataset_widget, category_widget, count_widget
])

# display(data_collection_widget)


device = torch.device('cuda')

# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, len(dataset.categories))

# SQUEEZENET 
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, len(dataset.categories), kernel_size=1)
# model.num_classes = len(dataset.categories)

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))

# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))
    
model = model.to(device)

model_save_button = ipywidgets.Button(description='save model')
model_path_widget = ipywidgets.Text(description='model path', value='/nvdli-nano/data/classification/my_model5.pth')

def load_model(c):
    global model, model_path_widget
    print("Load_model...")
    model.load_state_dict(torch.load(model_path_widget.value))
    print("Load_model!")

    
def save_model(c):
    torch.save(model.state_dict(), model_path_widget.value)
model_save_button.on_click(save_model)

model_widget = ipywidgets.VBox([
    model_path_widget,
    ipywidgets.HBox([model_save_button])
])



state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='state', value='stop')
prediction_widget = ipywidgets.Text(description='prediction')
score_widgets = []
for category in dataset.categories:
    score_widget = ipywidgets.FloatSlider(min=0.0, max=1.0, description=category, orientation='vertical')
    score_widgets.append(score_widget)

def live(state_widget, model, camera, prediction_widget, score_widget):
    global dataset
    while state_widget.value == 'live':
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        prediction_widget.value = dataset.categories[category_index]
        for i, score in enumerate(list(output)):
            score_widgets[i].value = score
            
def start_live(change):
    if change['new'] == 'live':
        execute_thread = threading.Thread(target=live, args=(state_widget, model, camera, prediction_widget, score_widget))
        execute_thread.start()

state_widget.observe(start_live, names='value')

live_execution_widget = ipywidgets.VBox([
    ipywidgets.HBox(score_widgets),
    prediction_widget,
    state_widget
])



BATCH_SIZE = 8

optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

epochs_widget = ipywidgets.IntText(description='epochs', value=epochs_to_train)
loss_widget = ipywidgets.FloatText(description='loss')
accuracy_widget = ipywidgets.FloatText(description='accuracy')
progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description='progress')

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, accuracy_widget, loss_widget, progress_widget, state_widget
    
    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        state_widget.value = 'stop'
        time.sleep(1)

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while epochs_widget.value > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
                progress_widget.value = i / len(dataset)
                loss_widget.value = sum_loss / i
                accuracy_widget.value = 1.0 - error_count / i
                
            if is_training:
                epochs_widget.value = epochs_widget.value - 1
            else:
                break
    except :
        pass
    model = model.eval()
    
    progress_widget.value = 0
    loss_widget.value = 0
    accuracy_widget.value = 0
    state_widget.value = 'live'
    
    
    
train_eval_widget = ipywidgets.VBox([
    epochs_widget,
    progress_widget,
    loss_widget,
    accuracy_widget
])

# display(train_eval_widget)

all_widget = ipywidgets.VBox([
    ipywidgets.HBox([data_collection_widget, live_execution_widget]), 
    train_eval_widget,
    model_widget
])


display(all_widget)

def update_image(change):
    global current_imagee 
    image = change['new']
    current_imagee = bgr8_to_jpeg(image)
    camera_widget.value = current_imagee

camera.observe(update_image, names='value')



class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global prediction_widget
        global current_imagee
        global epochs_widget
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(bytes(open("index.html","r").read(), "utf-8"))
            
        if self.path.startswith("/current_image"):
            self.send_response(200)
            self.send_header('Content-type','image/png')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(current_imagee)
            
        if self.path.startswith("/current_text"):
            self.send_response(200)
            self.send_header('Content-type','text/plain')
            self.send_header('Access-Control-Allow-Origin','*')

            self.end_headers()
            b64_string = base64.b64encode(current_imagee).decode('utf-8')
            self.wfile.write(bytes(b64_string, "utf-8"))
        if self.path == "/category/":
            self.send_response(200)
            self.send_header('Content-type','text/plain')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(bytes(str(prediction_widget.value), "utf-8"))
            
        if self.path == "/epochs/":
            self.send_response(200)
            self.send_header('Content-type','text/plain')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(bytes(str(epochs_widget.value), "utf-8"))
            
class WEB_SERVER(threading.Thread):
    def run(self,*args,**kwargs):
        global IP_address
        with HTTPServer(('', 8080), handler) as server:
            print("Web server started! http://jetson:8080/")
            server.serve_forever()
t = WEB_SERVER()
t.start()
if load_model_from_file:
        
    print("Please wait...")
    model.load_state_dict(torch.load(model_to_load))
    print("Done loading model!")

else:
    print("Please wait...")
    train_eval(True)
    print("Done learning!")
