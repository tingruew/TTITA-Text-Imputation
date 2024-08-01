# Text Imputation for Tabular Data
This repository provides a tool to impute missing textual data in your datasets.

## Setup
Create a conda environment with Python=3.11:
```
    conda create -n TTITA python=3.11
```
 
install the following (with the correct cuda version):
```
    conda install pytorch==2.2.0 torchtext==0.17.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install other dependencies in requirements.txt:
```
    pip install -r requirements.txt
```

## Demo
Take the toy review dataset below for example

|Rating|Reviewer ID|Item Description|Review Text|
|------|-----------|----------------|-----------|
|5|abcd|...|I like it!|
|1|efgh|...||
|3|abcd|...|It's ok, acceptable.|
|1|xyz|...|Won't buy it again...|

Code to impute the Review Text column
```
from impute import *

# Initialize the imputation model
imputer = Imputer({'Rating':'int',
                   'Reviewer ID':'cat',
                   'Item Description':'text'},
                   'Review Text')

# Fit the model
imputer.fit(train)

# Impute with the fitted model (test is a Pandas Dataframe containing the same columns)
test['predicted_review_text'] = imputer.predict(test)
```
See [demo.py](demo.py) for a more concrete example on car reviews

