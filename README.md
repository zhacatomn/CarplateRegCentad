# Documentation 



## Carplate Detection Files

```
model_v3.json
model_v3_3.h5
```

are the files containing the architecture and weights of the carplate detection model respectively. (`model_v3_3.h5` is too big, so it can’t be uploaded)

```
model_box.ipynb
```

is the Jupyter file where the data is processed, and where the model is trained and tested. 



## Character Detection Files

```
charloc3.py
```

is the file containing the algorithm which ascertains the position of each character in the carplate. It incudes both the contour detection, and the determining of the order of the characters.



## Character Recognition Files

```
charcollectedvgg.json
charcollectedvgg.h5
```

are the files containing the architecture and weights of the character recognition model respectively.

```
model_chardetect.ipynb
```

is the Jupyter file where the training and validation data is generated, and where the model is trained and tested. 



## Overall Framework Files

```
detect_plate1.py
```

is the python file which contains the function where the 3 sections of the framework are assembled together to fully predict the carplate number. 

```
detect_plate.ipynb
```

is the Jupyter file where the overall framework is tested.