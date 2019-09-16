# commtailment
A simulation framework for community-based curtailment of PV systems.

## Installation
have python 3.5 installed (3.6 raised issues with numba implementation,
which is important for massive speed up)
```
sudo apt-get install python3 python3-dev build-essentials
```

the use of virtualenv is recommended
```
virtualenv --python=/usr/bin/python3.5 env
```

activate environment
```
source env/bin/activate
```

install commtailment
```
pip install -r requirements.txt
pip install -e .
```
