import torch
import librosa
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from .utils import window_sumsquare
import numpy as  np
from pathlib import Path
from matplotlib import pyplot as plt

# 短时傅里叶变换(STFT)音频特征提取
class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann'):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).
        
        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris) 
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """Take input data (audio) to STFT domain.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        
        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced 
        by the ```transform``` function.
        
        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        
        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal

## 对谱的动态压缩，取对数
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

## 对谱的反动态压缩
def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

##  求梅尔谱 算法
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=11025):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        y = y.clamp(min=-1, max=1)
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
    
# 下面是提取 melspec的代码
def extract_mel_feature_bytaco(hp:Create_Prepro_Hparams):
    '''
    :param hp:
    将数据集的语音 保持原始目录结构提取到 另一个文件夹
    :return:
    '''
    # 提取mel谱
    stftfunc = TacotronSTFT(filter_length=hp.n_fft,###1024
                            hop_length=hp.hop_length,## 256
                            win_length=hp.win_length,###1024
                            n_mel_channels=hp.n_mels,### 80
                            sampling_rate=hp.sample_rate,###22050 different from tacotron2 22050
                            mel_fmin=hp.f_min,## 0.0
                            mel_fmax=hp.f_max)## 11025

    print("使用Taco2 mel 提取")

    ## 文件结构目录复制 ### 下面这段代码用path库实现， 是os库的加强版。 虽然看起来比较复杂。。 如果用os库则是这样写：
    '''
    # create the file struct in the new place, except the .wav
    for dirname, subdir, files in os.walk(datadir):# os.walk是获取所有的目录
            if dirname == datadir:
                pass
            else:
                dn = dirname.replace(datadir, out_dir)
                os.mkdir(dn)
    
    '''
    src_wavp = Path(hp.wav_datadir_name)
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(hp.wav_datadir_name,hp.feature_dir_name)).mkdir(parents=True,exist_ok=True)
    print("*" * 20)
    ########################################################################################
    wavpaths = [ x  for x in src_wavp.rglob('*.wav') if x.is_file() ]
    ttsum = len(wavpaths) # 总语音数量
    mel_frames = []
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp.resolve())
        the_melpath = str(wp.resolve()).replace(hp.wav_datadir_name,hp.feature_dir_name).replace('wav','npy')
        #wavform,_ = torchaudio.load(the_wavpath)
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=20)  ## 静音消除
        wavform = torch.FloatTensor(wavform).unsqueeze(0) ## [1,length]
        mel = stftfunc.mel_spectrogram(wavform)
        mel = mel.squeeze().detach().cpu().numpy()  # [mel_dim, frames]
        np.save(the_melpath, mel)

        ## 统计mel谱的长度 信息
        mel_frames.append(mel.shape[-1])
        print("{}|{} -- mel_length:{}".format(k,ttsum,mel.shape[-1]))

    mean_len = sum(mel_frames) / len(mel_frames)
    max_fl = max(mel_frames)
    min_fl = min(mel_frames)
    print("*" * 100)
    print("Melspec length , Mean:{},Max:{},min:{}".format(mean_len, max_fl, min_fl))

    pass

### 观察数据集的 时长分布，对于训练的效果有非常重要的意义。 因此我们观察提取的 全部的melspec的长度（帧数）
## 作出 直方图。
def plot_hist_of_meldata(datadirname):
    datadirp = Path(datadirname)

    mellens = []
    wavpaths = [ x for x in datadirp.rglob('*.npy') if x.is_file() ]

    for wavp in wavpaths:
        mel = np.load(str(wavp))
        mellens.append(mel.shape[-1])

    max_ = max(mellens)
    min_ = min(mellens)
    avg_ = int(sum(mellens)/len(mellens))
    print("max:{},min:{},avg:{}".format(max_,min_,avg_) )

    plt.figure()
    plt.title("MelLens_hist_" + "max:{},min:{},avg:{}".format(max_,min_,avg_))
    plt.hist(mellens)
    plt.xlabel("mel length")
    plt.ylabel("numbers")
    plt.savefig("Mel_lengths_hist" )
    plt.show()

if __name__=="__main__":
    # ################ preprocess  ###################################
    # # 将语音提取成meldata，用npy存储 （只存储 mel，无其他label）
    # wav_datadir_name = 'speaker_verify_dataset'
    # feature_dir_name = 'meldata_22k_trimed'
    # preprocess_hp = Create_Prepro_Hparams() # 创建参数类对象
    # preprocess_hp.set_preprocess_dir(wav_datadir_name,feature_dir_name) # 设置 源数据路径和目标路径
    # extract_mel_feature_bytaco(preprocess_hp) # 提取 mels特征


    ### 观察提取出来的melspec的时长分布图
    plot_hist_of_meldata("meldata_22k_trimed")
    '''
    上面程序的结果为：
    ****************************************************************************************************
    Melspec length , Mean:199.09066666666666,Max:375,min:41
    '''
