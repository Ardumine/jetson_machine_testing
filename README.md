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


### Installing

A step by step series of examples that tell you how to get a development env running.

Clone the repo: and cd to the directory

```
git clone https://github.com/Ardumine/jetson_machine_testing/
cd jetson_machine_testing
```

Now install

```
python3 installer.py

cd
chmod +x run_machine.sh
chmod +x jupiter.sh
```

##Setting up:


End with an example of getting some data out of the system or using it for a little demo.

## Usage

Add notes about how to use the system.
