import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

# tutorial 01

from bciflow.datasets.cbcic import cbcic

dataset = cbcic(subject=1, path='data/cbcic/')
# Este comando carrega o conjunto de dados para assunto 1 e armazena-o em um dicionário chamado dataset. 
# Garantir que o conjunto de dados esteja disponível em data/cbcic/ ou ajustar o caminho de acordo.

print(dataset["X"])
# x é um array NumPy que contém os sinais de EEG para cada amostra. ex.
# [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...] onde cada sub-array representa um sinal de EEG em um determinado tempo.
# Imprime os sinais EEG organizados como um array 4D:
# # ensaios: quantas repetições (épocas) da tarefa foram registradas
# # frequency_bands: para cada teste, os sinais são filtrados em diferentes bandas de frequência (se aplicável)
# # canais: cada eletrodo na tampa EEG usado
# # time_samples: o sinal EEG ao longo do tempo (em amostras)
# Exemplo de forma: (120, 1, 12, 4096) 120 ensaios, 1 banda de frequência, 12 eletrodos, 4096 amostras de tempo. 
# Se a frequência é 512Hz, significa que há 4096 amostras em 8 segundos


print(dataset["y"])
# Isso mostra uma lista de inteiros que representam a classe (ou tarefa) realizada em cada teste.
# Exemplo: [0, 0, 0, ..., 1, 1, 1]
# Cada número corresponde a uma tarefa mental (como mão esquerda, mão direita, etc.)

print(dataset["y_dict"])
# Isso imprime um dicionário mapeando números de classe para seu significado
# Exemplo de saída: {'left-hand': 0, 'right-hand': 1}
# Isso nos diz o que as classes 0 e 1 significam no dataset["y"].

print(dataset["events"])
# Isso mostra um dicionário contendo a marcação de tempo dos eventos~:
# onde cada evento é representado por um dicionário contendo informações como o tipo de evento, o tempo de início e a duração.

{'get_start': [0, 3],
 'beep_sound': [2],
 'cue': [3, 8],
 'task_exec': [3, 8]}
# Isso nos diz quando cada evento aconteceu (em segundos) durante a coleta de dados. Útil para segmentar os sinais em torno de eventos específicos

print(dataset["ch_names"])
# ch_names é uma lista de strings que contém os nomes dos canais de EEG.
# ex. ['C3', 'C4', 'O1', 'O2', ...]
# Cada nome representa uma localização física no limite do EEG.

print(dataset["sfreq"])
# Retorna a frequência de amostragem em Hz (por exemplo, 512.0). Isso nos diz quantas amostras por segundo foram registradas.

print(dataset["tmin"])
# Mostra a hora de início em segundos em relação aos marcadores de eventos (por exemplo, 0,0).
# Se fosse -1, isso indicaria que os dados começam 1 segundo antes do evento (útil para extração de linhas de base pré-evento).

