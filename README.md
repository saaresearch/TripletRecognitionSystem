# TripletNet recognition system

![img_2.png](img_2.png)

<details open>
<summary>Install</summary>

```console
git clone 
```
1. Install Python **>=3.6.5 and <3.7**
2. Create virtual environment:

```console
python -m venv .env
```

3. Activate environment
    - on windows: `.env\Scripts\activate`
    - on linux/mac: `source .env/bin/activate`

4. Install dependencies:

```console
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
</details>

## Description of the system
Our system allows you to train machine learning models on small datasets, and also allows you to get a ready-made quantized model for use on mobile devices

![](doc/image/img.png)

The project contains the following utilities:
* Preprocessing (`pdd/data_utils.py`)
* Utils for create dataloaders
   * Split datasetutils (`pdd/train_test_split.py`)
   * Prepare dataloaders utils(`train.py` and `pdd/data_utils.py` )
* Utils for training :
   * Training Triplet extactor model with KNN classifier (`train.py`, `pdd/trainer.py`, `pdd/model.py`, `pdd/tripletttorch.py`)
   * Training MLP classifier for extractor model (`classifier_train.py`)
* Utils for test model (`script.py`, `perceptron_script.py`,`confusion_matrix.py` )
* Utils for quantization
  * Static quantization available only for **MobileNetV2** model and training only on CPU (all files with prefix quant)
  * Dynamic quantization (`dynamic_quantization.py`)

### Usage example
The resulting model can be used for web and mobile applications:
#### Telegram bot:


<div align="center">

![img.png](doc/image/telegram.png)
</div>
<div align="center">
<a href="https://t.me/PlantDiseaseRecognitionBOT">
   <img src="doc/image/img_2.png" width="5%"/>
   </a>
<a href="https://github.com/WEBSTERMASTER777/telegrambot">
   <img src="doc/image/img_1.png" width="5%"/>
   </a>
</div>

#### Android Application

<div align="center">

![img.png](doc/image/android.png)
</div>
<div align="center">
<a href="https://drive.google.com/file/d/1xOYnELaa5x2cDNjoqUbgGbgXsb9DcaeJ/view?usp=sharing">
   <img src="doc/image/img_3.png" width="5%"/>
   </a>
<a href="https://github.com/WEBSTERMASTER777/AndroidRecognitionApp">
   <img src="doc/image/img_1.png" width="5%"/>
   </a>
</div>


## Custom Dataset and model training

Instruction for dataset loading   and git demo chek in <a href="https://colab.research.google.com/drive/1YhVJfTeCBbMov1Lo_V6XyzMnXP8rraPJ"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>






