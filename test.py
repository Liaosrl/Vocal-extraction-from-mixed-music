
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display
import soundfile as sf
from IPython.display import Audio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


n_fft=2048
num_frame=25
hop=n_fft//4
batch_size=16
sr=16000
scale=1.0
encuda=True
model_path='C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/win2048_25_log_norm1.pt'
testfile='C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/test/demo6.flac'
save_path='output_log_norm1.wav'

song, _ = librosa.load(testfile,sr=sr, mono=True)
spec_s, phase_s = librosa.magphase(librosa.stft(song,n_fft=n_fft, hop_length=hop))
ref=np.min(spec_s)
ref=1e-3
spec_s=20*np.log10((spec_s)/ref)
spec_s=np.maximum(spec_s,0)

num_epoch=np.size(spec_s,axis=1)//(num_frame)
input_s=np.zeros((num_epoch,n_fft//2+1,num_frame),dtype=float)

for epoch in range(num_epoch):
    input_s[epoch,:,:]=spec_s[:,epoch*num_frame:(epoch+1)*num_frame]

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.channel1 = 64
        self.channel2 = 128
        self.channel3 = 32
        self.kernel1 = 3
        self.kernel2 = 3
        pad1=1
        pad2=1

        self.convIn = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.channel1,
                      kernel_size=self.kernel2,
                      padding=pad2),
            nn.BatchNorm2d(self.channel1),          
            nn.ReLU()
        )
        
        self.convB = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel2,
                      kernel_size=self.kernel2,
                      padding=pad2),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()           
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()           
            #nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel2*2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
            #nn.MaxPool2d(2)
        )

        self.convOut = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1*2,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(),
            #nn.MaxUnpool2d(2),            
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel3,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel3,
                      out_channels=1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU()
        )
    def forward(self, input):

        input = input.unsqueeze(1)  # (batch, 1, freq, time)
        outputA = self.convIn(input)
        outputB = self.convB(outputA)
        output = self.conv1(outputB)
        output = self.conv2(torch.cat((outputB,output),1))
        output = self.convOut(torch.cat((outputA,output),1))
        return output[:,0,:,:]
    
model_CNN2D = CNN2D()

if encuda:
    model_CNN2D = model_CNN2D.cuda()
    model_CNN2D.load_state_dict(torch.load(model_path))
    print('previous CUDA model loaded')
else:
    model_CNN2D = model_CNN2D.cuda()
    model_CNN2D.load_state_dict(torch.load(model_path))
    model_CNN2D.cpu()
    print('previous model loaded')

model_CNN2D=model_CNN2D.eval()


batch_num=np.size(spec_s,axis=1)//(batch_size*num_frame)
input_tensor = torch.from_numpy(input_s.astype(np.float32))
if encuda:
    input_tensor = input_tensor.cuda()

for i in range(batch_num+1):   
    with torch.no_grad():
        if i==batch_num:
            model_output = model_CNN2D(input_tensor[i*batch_size:,:,:])
            
        model_output = model_CNN2D(input_tensor[i*batch_size:(i+1)*batch_size,:,:])

        if encuda:
            spec_output=model_output.cpu().detach().numpy()
        else:
            spec_output=model_output.detach().numpy()
    
    if i == 0:
        recover=spec_output
    else:
        recover=np.concatenate((recover,spec_output),axis=0)
    print(i)        

recover_o=recover*(test_max-test_min)/scale+test_min
recover_o=np.transpose(recover_o, (0, 2, 1)).reshape(-1, n_fft//2+1)
recover_o=np.transpose(recover_o)



recover_spec=librosa.db_to_amplitude(recover_o,ref=ref)
recover_spec=recover_spec-ref
res_magphase=recover_spec*np.exp(phase_s[:,0:np.size(recover_spec,axis=1)]*1j)
res_song=librosa.core.istft(res_magphase, hop_length=hop) 
print(recover_spec.shape)


plt.figure(figsize=(16, 10))
plt.subplot(2,1,1)
librosa.display.specshow(librosa.amplitude_to_db(recover_spec, ref=np.min), y_axis='log', x_axis='time', sr=sr)
plt.title('Extracted vocal')
plt.subplot(2,1,2)
librosa.display.specshow(spec_s, y_axis='log', x_axis='time', sr=sr)
plt.title('Original song')