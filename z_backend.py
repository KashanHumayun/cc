import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from flask import Flask, request, jsonify
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import numpy as np

# Define the model components
class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias) for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out

def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=2376, channels=512, embd_dim=192):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        return out

def load_model(model_path):
    model = ECAPA_TDNN(in_channels=2376)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Feature extraction functions
def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, n_mfcc=20, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    zcr_feature = zcr(data, frame_length, hop_length)
    rmse_feature = rmse(data, frame_length, hop_length)
    mfcc_feature = mfcc(data, sr, frame_length, hop_length, n_mfcc=20, flatten=False)
    
    # Ensure all features have the same number of frames
    num_frames = min(zcr_feature.shape[0], rmse_feature.shape[0], mfcc_feature.shape[1])
    zcr_feature = zcr_feature[:num_frames]
    rmse_feature = rmse_feature[:num_frames]
    mfcc_feature = mfcc_feature[:, :num_frames]
    
    # Flatten MFCC features and concatenate
    features = np.hstack((zcr_feature, rmse_feature, mfcc_feature.flatten()))
    
    # Ensure the final feature length is 2376
    if features.shape[0] < 2376:
        features = np.pad(features, (0, 2376 - features.shape[0]), 'constant')
    elif features.shape[0] > 2376:
        features = features[:2376]
    
    assert features.shape[0] == 2376, f"Expected feature length 2376 but got {features.shape[0]}"
    return features

# Data augmentation functions
def noise(data, noise_rate=0.01):
    noise_amp = noise_rate * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# Data augmentation function for the model
def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Raw data
    res1 = extract_features(data, sample_rate)
    
    # Data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    
    # Data with pitching
    data_pitch = pitch(data, sample_rate)
    res3 = extract_features(data_pitch, sample_rate)
    
    # Data with both pitching and noise
    data_noise_pitch = noise(data_pitch)
    res4 = extract_features(data_noise_pitch, sample_rate)
    
    # Combine features
    combined_features = np.vstack((res1, res2, res3, res4)).flatten()
    
    # Ensure combined_features has correct shape
    expected_size = 2376
    if combined_features.size < expected_size:
        combined_features = np.pad(combined_features, (0, expected_size - combined_features.size), 'constant')
    else:
        combined_features = combined_features[:expected_size]
    
    assert combined_features.size == expected_size, f"Expected feature vector of size {expected_size}, but got {combined_features.size}"
    return combined_features

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def predict_emotion(model, audio_path):
    features = get_features(audio_path)
    features = features.reshape(1, 1, -1)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output.data, 1)

    emotion_map = {0: 'disgust', 1: 'sad', 2: 'fear', 3: 'angry', 4: 'happy', 5: 'neutral', 6: 'surprise'}
    return emotion_map.get(predicted.item(), "Unknown")

app = Flask(__name__)
model = load_model('ECAPA_TDNN_audio_model.pth')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        if not file_path.endswith('.wav'):
            file_path = convert_to_wav(file_path)

        emotion = predict_emotion(model, file_path)
        return jsonify({'emotion': emotion})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
