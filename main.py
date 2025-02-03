import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Carregar o dataset BCI Competition IV 2A
# Supondo que o arquivo .mat esteja no mesmo diretório
data = scipy.io.loadmat('A01T.mat')

# Extraindo os sinais de EEG
eeg_data = data['data']  # Ajuste conforme a estrutura do seu arquivo .mat

# Definindo parâmetros
fs = 250  # Frequência de amostragem (ajuste conforme necessário)
n_channels = eeg_data.shape[1]  # Número de canais de EEG

# Aplicando a Transformada de Fourier em cada canal
eeg_fft = np.fft.fft(eeg_data, axis=0)
frequencias = np.fft.fftfreq(eeg_data.shape[0], d=1/fs)

# Plotando os resultados para o primeiro canal
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(eeg_data[:, 0])
plt.title('Sinal de EEG Original (Canal 1)')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frequencias, np.abs(eeg_fft[:, 0]))
plt.title('Transformada de Fourier (Canal 1)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()