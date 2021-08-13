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
from docxtpl import DocxTemplate, InlineImage
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


def features(signal, sampling_frequency):
    """Gives more attributes to the signal such as rr_interval, hr etc

    Parameters
    -----------
    signal : 
        Input ECG Signal

    sampling_frequency : int
        The Frequecny at which the signal is sampled
    
    Returns
    -------
    dataframe
        ecg_info-more attributes are added.
    """
    ecg_info = {}
    out = biosppy.ecg.ecg(signal=signal, sampling_rate=sampling_frequency, show=False)
    temp = biosppy.signals.ecg.extract_heartbeats(
        signal, rpeaks=out["rpeaks"], sampling_rate=sampling_frequency
    )
    df = pd.DataFrame(temp[0])
    pred = model.predict(df)
    rr_interval = np.array(
        [y - x for x, y in zip(out["rpeaks"][:-1], out["rpeaks"][1:])], dtype="float32"
    )
    ecg_info["rpeaks"] = out["rpeaks"]
    ecg_info["rr_interval"] = rr_interval
    ecg_info["prediction"] = pred
    ecg_info["heart_rate"] = out["heart_rate"].astype("int")
    return ecg_info


def plot(signal):
    """
    Parameters
    -----------
    signal : 
        Input ECG Signal

    Returns
    -------
    dataframe
        out-Performs ecg extraction from biosspy   
    """
    outs = ecg.ecg(signal=signal, sampling_rate=360, show=False)
    return outs


def standardise(signal, freq, adc):
    """Standardises the Signal to certain frequency and ADC Resolution
    Parameters
    -----------
    signal : 
        Input ECG Signal
    freq :
        The Required Frequency
    adc:
        The Required ADC Resolution
    Returns
    -------
    array
        Standarised signal with the specified frequency and adc resolution.
    """
    sig = resample_by_interpolation(signal, freq, 360).round(2)
    print(sig)
    sig = [((10 * x / 2 ** (adc) - 5).round(4)) for x in sig]
    return sig


def resample_by_interpolation(signal, input_fs, output_fs):
    """Resamples the SIgnal from current frequency to the requireed frequency
    Parameters
    -----------
    signal : 
        Input ECG Signal

    input_fs :
        Current Frequency

    output_fs:
        Required Frequency
    
    Returns
    -------
    array
        resampled_signal-New signal with the specified frequency.
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
    """Chunks the signals based on the input
    Parameters
    -----------
    seq : 
        sequence

    size :
        net size
    
    Returns
    -------
    array
        chunks of signals
    """

    chunks = [seq[i * size : (i * size) + size] for i in range(ceil(len(seq) / size))]
    if len(chunks[:-1]) < size:
        return np.array(chunks[:-1])
    else:
        return np.array(chunks)


def time_frequency(df):

    """Calculates the time frequency
    Parameters
    -----------
    df : 
        dataframe

    Returns
    -------
    array
        time frequecny by strippping
    """
    # cat=[]
    time_freq = []
    for strip in df:
        f, t, Zxx = signal.stft(strip, 360, nperseg=19, window="hamming", nfft=19)
        time_freq.append(Zxx)
    time_freq = np.array(time_freq)
    return time_freq


def prediction(signal):
    """Prediction of the Signal
    Parameters
    -----------
    signal : 
        Input ECG Signal

    Returns
    -------
    array
        pred1-Predictions of the ML Model.
    """
    chunk = chunks(signal, 1080)
    data = time_frequency(chunk)
    data = np.absolute(data)
    data = np.reshape(data, (data.shape[0], 10, 109, 1))
    pred1 = (model1.predict(data) > 0.8).astype("int32").ravel()
    pred1 = map(lambda x: "X" if x == 1 else "NN", pred1)
    pred1 = np.array(list(pred1))
    return pred1


def event(preds, rr, hr):

    """From the prediction we predict the events associated with the signal using rule based conditions.
    Parameters
    -----------
    preds : 
        Predicitons from ML Model
    rr:
        rr interval
    hr:
        heart rate
    Returns
    -------
    array
        event class associated with label.
    """
    event_class = []
    rr = rr
    hr = hr
    pred = preds
    if "V" in pred:
        event_class.append("Event is Ventricular")
        #         event_class + 'Sub events found:'
        if "NVNVN" in pred:
            event_class.append("Sub events found: Bigeminy")
        if "NNVNNVN" in pred:
            event_class.append("Sub events found: Trigeminy")
        if "VV" in pred:
            count1 = pred.count("VV")
            count2 = pred.count("VVV")
            if count2 - count1 == 0:
                event_class.append("Sub events found: Triplet")
            elif count2 == 0:
                event_class.append("Sub events found: Couplet")
            else:
                event_class.append("Sub events found: Triplet,Couplet")
                # event_class.append('Sub events found: Couplet')

    elif all(ele < 60 for ele in hr):
        event_class.append("Event is Bradycardia")
        if all(ele < 60 and ele > 50 for ele in hr):
            event_class.append(
                "Bradycardia, Sub events found: Sinus Bradycardia, 50-60"
            )
        if all(ele < 40 for ele in hr):
            event_class.append("Bradycardia, Sub events found: Rate < 40")
        if any(ele >= 900 for ele in rr):
            event_class.append("Bradycardia, Sub events found: Pause> 2.5 sec")

    else:
        event_class.append("Event is Supraventricular")
        if all(ele >= 60 and ele <= 100 for ele in hr):
            if ((rr.max() - rr.min()) / rr.max()) * 100 > 10:
                event_class.append("Sub events found: Sinus Arrhythmia")
            else:
                event_class.append("Sub events found: Normal Sinus Rhythm")

        if all(ele > 100 for ele in hr):
            event_class.append("Sinus Tachycardia")
            if all(ele > 100 and ele < 150 for ele in hr):
                event_class.append("Sinus Tachycardia, 100-150")
            if all(ele > 150 for ele in hr):
                event_class.append("Sinus Tachycardia > 150")
        # else:
        # event_class.append('Sub event: Not defined for now')
    return event_class


