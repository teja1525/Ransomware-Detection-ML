import numpy as np
import joblib

import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import pandas as pd
import numpy as np
import pickle
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime

warnings.filterwarnings('ignore')



def predict():
    int_features= [21506087	,8530,	869836,	58991,	3,	4,	24576,	44,	1384448,	0,	462746,	10921356,	0]
    
    final4=[np.array(int_features)]
    model = joblib.load('model1.sav')
    predict = model.predict(final4)

    if predict == 0:
        output='Benign!'

    elif predict == 1:
        output = 'Ransomware!'
    

    return output

Result = predict()
print(Result)

