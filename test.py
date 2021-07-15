import os

from sklearn.preprocessing import minmax_scale
from tensorflow.keras.models import load_model
from pymedicompecg import procedure_builder as pb

import joblib
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import resample
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.models import load_model
from pymedicompecg import procedure_builder as pb


def test_beats():
    folder, _ = os.path.split(pb.__file__)
    noise_model = joblib.load(folder + os.sep + "models/RF_noise.pkl")
    beat_model = load_model(folder + os.sep + "models/AHA-model-dummy.h5")

    ecg = electrocardiogram()
    # resample to 250, originally set to 36
    ecg = resample(ecg, 250 * 300)
    ecg_scaled = minmax_scale(ecg, feature_range=(0, 2 ** 12)).astype(np.uint16)

    proc = pb.ProcedureBuilder(
        ecg_scaled, int(ecg_scaled.size / 250), noise_model, beat_model
    )

    proc.set_rpeaks()

    proc.set_beats()
    proc.set_ecg(ecg_scaled)


test_beats()


jjjfajfakfkkafka
