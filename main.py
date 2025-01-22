import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sci
import scipy.integrate as integrate

'''
x = np.random.random(100)
print(x)
plt.plot(x)
plt.show()

def function(x):
    return x

r = integrate.quad(function, 0, 10)
print(r)
plt.show()

'''

# Definindo o sinal de exemplo
t = np.linspace(0, 1, 500)  # Intervalo de tempo de 0 a 1 segundo, com 500 pontos
freq = 5  # Frequência do sinal em Hz
sinal = np.sin(2 * np.pi * freq * t)  # Sinal senoidal

# Aplicando a Transformada de Fourier
sinal_fft = np.fft.fft(sinal)
frequencias = np.fft.fftfreq(len(sinal), d=t[1] - t[0])

# Plotando o sinal original
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, sinal)
plt.title('Sinal Original')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')

# Plotando a Transformada de Fourier
plt.subplot(2, 1, 2)
plt.plot(frequencias, np.abs(sinal_fft))
plt.title('Transformada de Fourier')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()