import numpy as np
from numpy import asarray
import pandas as pd
from io import StringIO
import shutil
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage import data
from skimage.segmentation import slic
from skimage.util import img_as_float
from PIL import Image
import base64
from bson.objectid import ObjectId
from flask_pymongo import PyMongo
import scipy.io
import io
from flask_appbuilder.models.mixins import ImageColumn
import biosppy
import joblib
import re
from flask import Flask, render_template, redirect
from pymongo import MongoClient
from bson import Binary
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, MultipleFileField
from wtforms.validators import DataRequired
import plotly.graph_objects as go
import plotly.express as px
import plotly
from math import ceil
from tensorflow.keras.models import load_model
from scipy import signal
from biosppy.signals import ecg
from biosppy.plotting import plot_ecg
from fpdf import FPDF
from docxtpl import DocxTemplate,InlineImage
import jinja2
import docx2pdf
from docx2pdf import convert 
from docx.shared import Mm
import re
import itertools
from math import ceil
import numpy as np
from scipy import signal
import biosppy
from biosppy.signals import ecg
from biosppy.plotting import plot_ecg
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


"""
    This file contains the code for running ECG Flask Application made for Relfections Info Syste Pvt Ltd. The Codes are written in
    Python and various helper functions are written. Given a ECG Signal sampled at 360 Hz, this flask application gives the Prediction 
    that is in the Signal. A CNN Noise Classifier and ML Beat Model is also here along with the package. The plots are interactive in 
    nature and is done using plotly library. A PDF report will be generated in the end.
"""


def features(signal,sampling_frequency):
    """Gives more attributes to the signal such as rr_interval, hr etc

    Args:
        signal : Input ECG Signal
        sampling_frequency (int): The Frequecny at which the signal is sampled
    Returns:
        ecg_info: more attributes are added.
    """
    ecg_info={}
    out = biosppy.ecg.ecg(signal=signal, sampling_rate=sampling_frequency, show=False)
    temp=biosppy.signals.ecg.extract_heartbeats(signal, rpeaks=out['rpeaks'],sampling_rate=sampling_frequency)
    df=pd.DataFrame(temp[0])
    pred=model.predict(df)
    rr_interval=np.array([y-x for x, y in zip(out['rpeaks'][:-1], out['rpeaks'][1:])], dtype='float32')
    ecg_info['rpeaks']=out['rpeaks']
    ecg_info['rr_interval'] = rr_interval
    ecg_info['prediction']=pred
    ecg_info['heart_rate']=out['heart_rate'].astype('int')
    return ecg_info 


def plot(signal):
    """

    Args:
        signal : Input ECG Signal

    Returns:
        out : Performs ecg extraction from biosspy
    """
    outs = ecg.ecg(signal=signal, sampling_rate=360, show=False)
    return outs

def standardise(signal,freq,adc):
    """Standardises the Signal to certain frequency and ADC Resolution
    Args:
        signal: Input ECG Signal
        freq  : The Required Frequency
        adc   : THe Required ADC Resolution
    Returns:
        sig   : Standarised signal with the specified frequency and adc resolution.
    """
    sig=resample_by_interpolation(signal, freq, 360).round(2)
    print(sig)
    sig=[((10*x/2**(adc)-5).round(4)) for x in sig]
    return sig

def resample_by_interpolation(signal, input_fs, output_fs):
    """Resamples the SIgnal from current frequency to the requireed frequency

    Args:
        signal    : Input ECG Signal
        input_fs  : Current Frequency
        output_fs : Required Frequency

    Returns:
        resampled_signal: New signal with the specified frequency.
    """

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def chunks(seq, size):
    """[summary]

    Args:
        seq ([type]): [description]
        size ([type]): [description]

    Returns:
        [type]: [description]
    """
    chunks = [seq[i * size:(i * size) + size] for i in range(ceil(len(seq) / size))]
    if len(chunks[:-1])<size:
        return np.array(chunks[:-1])
    else:
        return np.array(chunks)

def time_frequency(df):
    """

    Args:
        df : Dataframe

    Returns:
        time_freq : time frequecny by strippping
    """
    #cat=[]
    time_freq=[]
    for strip in df :
        f, t, Zxx = signal.stft(strip, 360,nperseg=19,window='hamming',nfft=19)
        time_freq.append(Zxx)
    time_freq=np.array(time_freq)
    return time_freq

def prediction(signal):
    """Prediction of the Signal

    Args:
        signal : Input ECG Signal
    Returns:
        pred1  : Predictions of the ML Model.
    """
    chunk=chunks(signal,1080)
    data=time_frequency(chunk)
    data=np.absolute(data)
    data=np.reshape(data, (data.shape[0], 10, 109, 1))
    pred1=(model1.predict(data) > 0.8).astype("int32").ravel()
    pred1 = map(lambda x: 'X' if x==1 else 'NN', pred1)
    pred1=np.array(list(pred1))
    return pred1

def event(preds,rr,hr):
    """From the prediction we predict the events associated with the signal using rule based conditions.

    Args:
        preds : Predicitons from ML Model
        rr : rr interval
        hr : heart rate
    """
    event_class = []
    rr=rr
    hr=hr
    pred = preds
    if 'V' in pred :
        event_class.append('Event is Ventricular')
#         event_class + 'Sub events found:'
        if 'NVNVN' in pred:
            event_class.append('Sub events found: Bigeminy')
        if 'NNVNNVN' in pred:
            event_class.append('Sub events found: Trigeminy')
        if 'VV'in pred:
            count1=pred.count('VV')
            count2=pred.count('VVV')
            if count2-count1==0:
                event_class.append('Sub events found: Triplet')
            elif count2==0:
                event_class.append('Sub events found: Couplet')
            else:
                event_class.append('Sub events found: Triplet,Couplet')
                #event_class.append('Sub events found: Couplet')
        
    elif all(ele < 60 for ele in hr):
        event_class.append('Event is Bradycardia')
        if all(ele < 60 and ele>50 for ele in hr):
            event_class.append('Bradycardia, Sub events found: Sinus Bradycardia, 50-60')
        if all(ele < 40 for ele in hr):
            event_class.append('Bradycardia, Sub events found: Rate < 40')
        if any(ele >=900 for ele in rr):
            event_class.append('Bradycardia, Sub events found: Pause> 2.5 sec')
    
    else:
        event_class.append('Event is Supraventricular')
        if all(ele>=60 and ele<=100 for ele in hr):
            if ((rr.max() - rr.min())/rr.max())*100>10:
                event_class.append('Sub events found: Sinus Arrhythmia')
            else:
                event_class.append('Sub events found: Normal Sinus Rhythm')
        
        if all(ele>100 for ele in hr):
            event_class.append('Sinus Tachycardia')
            if all(ele>100 and ele<150 for ele in hr):
                event_class.append('Sinus Tachycardia, 100-150')
            if all(ele>150 for ele in hr):
                event_class.append('Sinus Tachycardia > 150')
        #else:
            #event_class.append('Sub event: Not defined for now')
    return event_class

damage_folder = os.path.join('static', 'ecg')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = damage_folder

client = MongoClient() 
db = client['ecgsampledb']
coll = db['json-config']

app.config['SECRET_KEY'] = 'mySecretKey'

model1=load_model(r"D:\Medicomp\documentation\Testing\ECG_Reflection-main\Noise_Classifier_360_3s.h5")

model =joblib.load(r"D:\Medicomp\documentation\Testing\ECG_Reflection-main\SVC_Model_5_new.pkl")

class JSONUploadForm(FlaskForm):
    choose_file = MultipleFileField('Choose your file', validators=[DataRequired()])
    submit = SubmitField('Submit') 
    

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = JSONUploadForm()
    if form.validate_on_submit():
        if os.path.isdir('static'):
            shutil.rmtree('static')
            os.mkdir('static')
            os.mkdir('static/images')
        else:
            os.mkdir('static') 
            os.mkdir('static/images')
        abc = 1
        b64_list=[] 
        global concat 
        concat = np.array([])
        global hrv
        hrv = np.array([])
        global beats
        beats = np.array([])
        global indexrange
        indexrange = []
        global eventlabel
        eventlabel = []
        global eventlist
        eventlist = [] 
        for file in form.choose_file.data:
            print(type(file))
            data=json.load(file) 
                        
            if 'signal' in data.keys():
                #data = eval(file.read().decode('utf-8')) 
                target_format = dict()
                target_format['signal'] = data
                file1 = np.array(data['signal'])

            else:
                sig=list(map(int,data['ECGDataMessageClass']['CDataChannelA'].split(' ')))
                sig=standardise(sig,int(data['ECGDataMessageClass']['SamplingFrequency']),int(re.findall("\d+",data['ECGDataMessageClass']['ADCResolution'] )[0]))
                target_format = dict()
                target_format['signal'] = sig            
                file1 = np.array(sig)
            indexrange.append(len(file1))
            concat = np.concatenate((concat,file1))
            feature = features(file1, 360)
            pred = feature['prediction']
            rr = feature['rr_interval']
            hr = feature['heart_rate']
            outs = plot(file1)
            plt.rcParams["figure.figsize"] = (15,6)
            plot_ecg(ts=outs['ts'],raw=file1,filtered=outs['filtered'],rpeaks=outs['rpeaks'],templates_ts=outs['templates_ts'],templates=outs['templates'],heart_rate_ts=outs['heart_rate_ts'],heart_rate=outs['heart_rate'],path='static/images/file{}'.format(abc)  + '.png',show=False)
            
            preds = "".join(pred)
            events = event(preds,rr,hr)
            eventlist.append(events) 
            hr=np.round(hr)
            print("Predicted Event class: ", events)
            title = events
            title = ", ".join(title)
            x = feature['rpeaks']
            y = file1[feature['rpeaks']]
            
            predict=prediction(file1)
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x,y=y, mode="markers+text",text=pred,name="Rpeaks",textposition="top center",marker=dict(color='#fc0505'), textfont=dict(color="#fc0505")))
            fig.add_trace(go.Scatter(x=x,y=y, mode="markers+text",text=hr,name="Heart Rate",textposition="top right",marker=dict(color='#aaad0a'), textfont=dict(color="#aaad0a")))
            fig.add_trace(go.Scatter( x=np.arange(0,len(file1)), y=file1, name="Signal", textposition="top center", marker=dict(color='#00cf7c')))
            fig.update_layout(paper_bgcolor='#173048',plot_bgcolor='#173048',font_color="#aaad0a",title=title)
            
            events=(" ".join(repr(e) for e in events))
            if 'Sinus Tachychardia' in events or 'Sinus bradycardia' in events or 'Sinus Arrhythmia' in events:
                eventlabel.append("#ff8800")
            elif 'Ventricular' in events or 'bradycardia' in events:
                eventlabel.append('#fc0303')
            else:
                eventlabel.append('#173048') 
            beats = np.concatenate((beats,pred))
            hrv = np.concatenate((hrv,hr)) 
# =============================================================================
#             print(len(hr))
#             print(len(pred))
#             print(len(rr))
# =============================================================================
            
            ind=np.where(predict =='X')[0]
            print("length of ind", ind)
            for i in ind:
                ind1=1080*(i)
                ind2=1080*(i+1) 
                fig.add_trace(go.Scatter(x=[ind1,ind1,ind2,ind2], y=[5,-5,-5,5], fill='toself', fillcolor = "#807f7e",
                    hoveron='points', line_color="#807f7e",
                    text="Points only",opacity=1,hoverinfo='text+x+y',mode='none'))
                fig.add_trace(go.Scatter( x=np.arange(ind1,ind2), y=file1[ind1:ind2],name="Signal", marker=dict(color='#00cf7c'), textposition="top center"))
                fig.update_layout(showlegend=False)
            #fig.update_layout(autosize=False, width=500, height=200, margin=dict(l=50,r=50,b=50,t=50,pad=4),paper_bgcolor="LightSteelBlue",)
            
            #fig.show()
            #fig.write_image('temp/file{}'.format(abc)  + '.png')
            print("fig type is:", type(fig))
            plotly.offline.plot(fig, filename='static/file{}'.format(abc)  + '.html', auto_open=False, include_plotlyjs="cdn") 
            
            img_list=glob.glob('static/*.html')
            print(rr)
        
            b64_list.append('../static/file{}'.format(abc)  + '.html') 
            
            abc+=1
        beats = "".join(beats.ravel())
        beats = dict(zip(beats,[beats.count(i) for i in beats]))
            
        return render_template('index.html', count = len(b64_list)) 
    return render_template('demolayout.html', upload_form=form) 

@app.route('/page', methods=['GET', 'POST'])
def con():   
    feature = features(concat, 360)
    predss = feature['prediction']
    rr = feature['rr_interval']
    hr = feature['heart_rate']
    
    eventss = event(predss,rr,hr)
    print("Heart rate variability average: ",np.mean(hrv).round() )
    print("Beats predicted: ", beats)
    print("Event list: ", eventlist)
    predsss = "".join(predss)
    print("predictions", predsss)
    x = feature['rpeaks']
    y = concat[feature['rpeaks']]   
    
    eventlistt = list(itertools.chain(*eventlist))
    string = " ".join(eventlistt)
    substring = "Ventricular"
    v1 = string.count(substring)
    substring = "Tachycardia"
    v2 = string.count(substring)
    substring = "Triplet"
    v3 = string.count(substring)
    substring = "Couplet"
    v4 = string.count(substring)
    substring = "Bigeminy"
    v5 = string.count(substring)
    substring = "Trigeminy"
    v6 = string.count(substring)
    substring = "Bradycardia"
    v12 = string.count(substring)
    
    fig = px.line(x=np.arange(0,len(concat)), y=concat, title='ECG Strips Combined', color_discrete_map={"Average": "#fc0d0d","Raw": "#fc0d0d"})
    fig.add_trace(go.Scatter(x=x,y=y, mode="markers+text",text=predss,name="Rpeaks",textposition="top center",marker=dict(color='#fc0505'), textfont=dict(color="#fc0505")))
    fig.add_trace(go.Scatter(x=x,y=y, mode="markers+text",text=hr,name="Heart Rate",textposition="top right",marker=dict(color='#aaad0a'), textfont=dict(color="#aaad0a")))
    fig.add_trace(go.Scatter( x=np.arange(0,len(concat)), y=concat, name="Signal",  marker=dict(color='#00cf7c'), textposition="top center"))
    fig.update_layout(paper_bgcolor='#173048',plot_bgcolor='#173048',font_color="#aaad0a")
    print("indexrange is ", indexrange)
    print("eventlabel is ", eventlabel)
    j = 0
    k = 0
    l = 1
    x = indexrange[k]
    for i in range(len(indexrange)):
        try:
            print(j,x)
            fig.add_vrect(x0=j, x1=x, fillcolor=eventlabel[k], opacity=0.4, layer="below", line_width=0)
            j = x
            x = x + indexrange[l]
            l+=1
            k+=1
        except IndexError:
            pass
    
    predict=prediction(concat)
    ind=np.where(predict =='X')[0]
    for i in ind:
        ind1=1080*(i)
        ind2=1080*(i+1)
        fig.add_trace(go.Scatter(x=[ind1,ind1,ind2,ind2], y=[6,-6,-6,6], fill='toself', fillcolor = "#807f7e",
                    hoveron='points', line_color="#807f7e",
                    text="Points only",opacity=1,hoverinfo='text+x+y',mode='none'))
        fig.add_trace(go.Scatter( x=np.arange(ind1,ind2), y=concat[ind1:ind2],name="Signal", marker=dict(color='#00cf7c'), textposition="top center"))
        fig.update_layout(showlegend=False)
           
    
# =============================================================================
#     ind = 0
#     indd = 215
#     for i in predss:
#         print(i)
#         if i == "N":
#             fillcolor = '#0aa302'
#         elif i == "V":
#             fillcolor='#cc0213'
#         elif i == "L" or i == "R":
#             fillcolor='#fc7405'
#         else:
#             fillcolor='#fcf803'
#         fig.add_vrect(x0=ind, x1=indd,fillcolor=fillcolor, opacity=0.4,layer="below", line_width=0)
#         ind+=215
#         indd+=215
# =============================================================================
            
    print("fig type is:", type(fig))
    plotly.offline.plot(fig, filename='static/concat.html', auto_open=False, include_plotlyjs="cdn")    
    fig.write_image("concatt.png") 
    
    y={'N':'Normal Beat','V':'Premature ventricular contraction','L':'Left bundle branch block beat','R':'Right bundle branch block beat','/':'Paced Beat'}
    labels=[y[key] for key in beats.keys()] 
    fig = go.Figure(data=[go.Pie(labels=labels, values=list(beats.values()), textinfo='label+percent',insidetextorientation='radial')])
    fig.update_layout(autosize=False,width=450,showlegend=False,height=250,margin=dict(l=0, r=0, b=0, t=0, pad=1))
    
    fig.write_image("piechart.png")
    plotly.offline.plot(fig, filename='static/piechart.html', auto_open=False, include_plotlyjs="cdn") 

    rr = round((rr/360).max(),3)
    if rr>2.5:
        pause = 'Yes'
    else:
        pause = 'No'
    length=len(concat)/360      
    doc=DocxTemplate('edit1.docx')
    context={'beats':beats,
         'mean':(np.mean(hrv).round()),
         'events':eventss,
         'min':min(hrv),
         'max':max(hrv),
         'len':length,
         'rr':rr, 'v1':v1, 'v2':v2, 'v3':v3, 'v4':v4, 'v5':v5, 'v6':v6, "v12":v12,
         'pause':pause,
         'png': InlineImage(doc, 'piechart.png', width=Mm(70),height=Mm(60))}
    doc.render(context)
    doc.save('1234.docx')
    import pythoncom
    pythoncom.CoInitialize()
    convert('/static/1234.docx','/static/report.pdf')
    os.remove("1234.docx")  
    return render_template('concat2.html', beats=beats, hrv=np.mean(hrv).round(),eventss=eventss)

if __name__ == "__main__":
    app.run()
    
