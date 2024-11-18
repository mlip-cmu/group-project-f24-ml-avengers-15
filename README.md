# group-project-f24-ml-avengers-15
group-project-f24-ml-avengers-15 created by GitHub Classroom


# Movie Recommender System

## Setup

### Prerequisites

- Python 3.9 or above (3.11 recommended)
- Open in VSCode and install recommmended extensions for optimal dev experience
- sudo apt install python3.x-venv (For LINUX)

### Create virtual environment

```bash
python -m venv .venv
```

### Activate virtual environment

```bash
#UNIX
source .venv/bin/activate

#Windows
.venv/Scripts/activate.bat
#OR Powershell
.venv/Scripts/Activate.ps1
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Create .env file and set required properties

```bash
cp example.env .env
```
Five items will need to be configured in .env:
* VM Server IP: insert your vm ip address (`TEAM_15_SERVER_IP`)
* Kafka Server IP: insert your kafka stream server ip address (`SERVER_IP`)
* SSH User: ssh username (`SSH_USER`)
* SSH Password: ssh password (`SSH_PASSWORD`)
* Kafka Port: port at which kafka stream is running (`KAFKA_PORT`)
* Local Port: local machine server listening for kafka stream (`LOCAL_PORT`)

## Running the Application

### Start the Server

```bash
python3 app.py
```

The server will start on `http://localhost:8082` by default.

## Features

### Experiments Dashboard

The application includes a comprehensive A/B testing infrastructure for evaluating and comparing different recommendation models. Access the experiments dashboard at `http://localhost:8082/experiments`.

#### Available Features:

1. **Create New Experiments**
   - Define experiment name
   - Select Model A and Model B for comparison
   - Set traffic split between models
   - Configure experiment parameters

2. **Monitor Active Experiments**
   - Real-time performance metrics
   - Statistical significance testing
   - Visualizations for:
     * Accuracy distributions
     * Latency measurements
   - Sample size tracking for both models

3. **Experiment Management**
   - End running experiments
   - Delete completed experiments
   - View historical performance data

4. **Performance Metrics**
   - Accuracy metrics with confidence intervals
   - Latency measurements with standard deviation
   - Effect size calculations
   - P-value significance testing

### Best Practices

1. **Creating Experiments**
   - Use descriptive names for easy identification
   - Start with a 50-50 traffic split for unbiased testing
   - Ensure both models are properly configured before starting

2. **Monitoring**
   - Wait for sufficient sample size before drawing conclusions
   - Check both accuracy and latency metrics
   - Consider statistical significance (p-value < 0.05) when making decisions

3. **Ending Experiments**
   - Document findings before ending experiments
   - Ensure all necessary data is collected
   - Consider running follow-up experiments if results are inconclusive
