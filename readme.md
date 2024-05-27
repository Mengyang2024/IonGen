# IonGen: An organic ion generator
IonGen is a repository for generating de novo organic ions. Please cite:

### Install envirement:
```shell
conda create -n IonGen python=3.8
conda activate IonGen
conda install cudatoolkit=11.8.0
conda install cudnn=8.9.2.26 -c anaconda
pip install tensorflow==2.13.0 rdkit==2023.9.4 pandas==1.5.3 graphviz
```

### To generate de novo cations:
```shell
python gen_cations.py
```

### To generate de novo anions:
```shell
python gen_anons.py
```

