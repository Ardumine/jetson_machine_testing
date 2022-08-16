# Jetson machine testing

(Sorry for bad english. Im trying my best :))

## Table of Contents
+ [About](#about)
+ [Getting Started](#getting_started)
+ [Usage](#usage)

## About
This project was inpired from my dad, wich he makes AI for choosing cork stoppers (I think thats how you say it), and i decided to make a project with a Jetson, so he seletcts the output of a machine. 

In this case, the output of the machine will two different types of cans. Some of them werent done correclty and have a extra solder(I used sticks becuse I dont have a solder) on top atached to it and 2 of them were done correctly, with the solder being cut(see in [here](images/jetson.png)). 

I was going to use 3 cans, but Jetson crashes when trying to get him to learn. But with just two cans, I got it right. In this case the 2 cans are 4 different models.


## Getting Started 
To get sarted we just need the Jetson itself, with jetpack installed, with an ethernet connection, with a ssh session.

### Prerequisites

Git: To install use:
```
sudo apt install git
```


## Installing


Clone the repo: and cd to the directory

```
git clone https://github.com/Ardumine/jetson_machine_testing/
cd jetson_machine_testing
```
Now run the installer
```
python3 installer.py

cd
chmod +x run_machine.sh
chmod +x jupiter.sh
```

## Setting up the software
There are two modes to use models: 
* The learn on start 
* Load model on start

### To use the learn on start
First copy the data to this dir:
```
~/nvdli-data/classification/
```
The data you should have in that folder should be like [this](data_trained/images/).

Edit the code file:
```
nano ~/nvdli-data/machine/main.py
```


Now go to this line:
```
###################CONFIG
```
Here is the config of the program. 
Change this line:
```
load_model_from_file = False
```
Now change cans_3 with your model name.
You can set epochs_to_train to more, but it will take much longer
```
#############LEARN FROM SCRATCH ON START CONFIG

TASK = 'cans_3' #Here put your folder name in classification, without the _A
epochs_to_train = 3 
```

### To use the load model on start
First copy the model to this dir:
```
~/nvdli-data/classification/
```
The model you should have in that folder should be like [this](data_trained/model/).

Edit the code file:
```
nano ~/nvdli-data/machine/main.py
```


Now go to this line:
```
###################CONFIG
```
Here is the config of the program. 
Change this line:
```
load_model_from_file = True
```
Now change model_to_load end to the model name. Dont forget to change the CATEGORIES to your categories.

```
#############LOAD MODEL ON START CONFIG
CATEGORIES = [ '7_up_broken', '7_up_ok', 'icetea_broken', 'icetea_ok'] #our catagories, that our ai will identify 

model_to_load = "/nvdli-nano/data/classification/my_model5.pth" # here put the path of your model. Make sure its of the docker conatiner.
```



## Running
```
./run_machine.sh
```

# INFO!

