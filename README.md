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
