# IonGen: An organic ion generator

IonGen is a repository for generating de novo organic ions utilizing a combination of Monte Carlo tree search and recurrent neural network techniques
![image](https://github.com/Mengyangjp/IonGen/assets/127812221/a9abe528-16c9-441c-9e7c-7f82299f2020)

## Usage

### Installation:
Clone the repository and install the required packages:
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

### Generated ions:

Virtual molecular library of 1 million generated cations and 1 million generated anions, named IonGen_1M, can be found at this link: https://figshare.com/articles/dataset/IonGen_1M/26343901

## Reference

If you find the code useful for your research, please consider citing

```
@inproceedings{Mengyang_2024_IonGen,
  author       = {Mengyang Qu, Gyanendra Sharma, Naoki Wada, Hisaki Ikebata, Shigeyuki Matsunami and Kenji Takahashi},
  title        = {Machine Learning-Driven Generation and Screening of Potential Ionic Liquids for Cellulose Dissolution},
  year         = 2024,
  publisher    = {},
  howpublished = {},
}
```
