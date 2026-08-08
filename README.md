## Setup

### Create a Python virtual environment

```bash
python3 -m venv venv
```

### Activate the virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python3 -m pip install -U "bitsandbytes>=0.46.1"
pip3 install -r requirements.txt -v
```

### Run

```bash
python3 submission.py
```

### Deactivate the virtual environment

```bash
deactivate
```
