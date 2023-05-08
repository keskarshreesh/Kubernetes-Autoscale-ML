## Kubernetes ML-Autoscaler

This is an implementation which trains and deploys an ML based autoscaler for Kubernetes
The setup instructions are valid for Linux environments and for Windows with WSL (Windows Subsystem for Linux).

## Initial Setup
1. git clone https://github.com/keskarshreesh/Kubernetes-Autoscale-ML.git
2. cd ms-deploy
3. git clone https://github.com/ewolff/microservice-kubernetes.git

## Setup Python Virtual Environment
Required: Python >= v3.6
1. Change directory to project root
2. virtualenv autoscaleml
3. source autoscaleml/bin/activate
4. pip install -r requirements.txt

## Deploy microservice:
1. Install kubectl
2. Install and start Docker Desktop (Windows) OR Install and start minikube (Linux/Windows, by running: minikube start --memory=4000)
3. Run kubernetes-deploy.sh

## Run Load Test
python loadtesting.py --name test_name --load 350 -s 6 -r 1 -c "1,2,3,4,5,6" -H "../data/history.csv" -S "../data/raw.csv" --locust

With default settings, the load test will run for 38 minutes.

## Run Model Training
python train_models.py --mode=train
Models will be saved to models directory, will take about 5 minutes each to train.

## Run Autoscaler evaluation
python custom_autoscaler.py
Evaluation results will be printed to the console
