#import os, sys
#project_directory = os.getcwd()
#sys.path.append(project_directory)

import sys
sys.path.append(r'C:\Users\Bernardo\Documents\Python\tcc\bciflow')

from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

# tutorial 01 exemplo

from bciflow.datasets.cbcic import cbcic
# Estamos utilizando o conjunto de dados CBCIC (Desafio Clínico de Interface Cérebro-Computador).

dataset = cbcic(subject=1, path='data/cbcic/') 
# Em seguida, carregando os dados.

from bciflow.modules.tf.filterbank import filterbank

#pre_folding = {'tf': {filterbank, {'kind_bp': 'chebyshevII'}}}
pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}
# Para replicar o algoritmo FBCSP, primeiro inicia o processo de dados utilizando um filterbank para aplicar múltiplos 
# filtros passa-faixa e capturar padrões em diferentes bandas de frequência.

pre_folding['tf'][0]  # filterbank
pre_folding['tf'][1]  # {'kind_bp': 'chebyshevII'}

from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

sf = csp()
fe = logpower
fs = MIBIF(8, clf=lda())
clf = lda()

#pos_folding = {
#    'sf': {sf, {}},
#    'fe': {fe, {}},
#    'fs': {fs, {}},
#    'clf': {clf, {}}
#}
pos_folding = {
    'sf': (sf, {}),
    'fe': (fe, {}),
    'fs': (fs, {}),
    'clf': (clf, {})
}
pos_folding['sf'][0]  # sf
pos_folding['sf'][1]  # {}
# Depois disso, a próxima etapa adiciona, em ordem, as fases do algoritmo:
## sf: Common Spatial Patterns (CSP) - maximiza a variância discriminativa
## fe: logpower - extrai a potência logarítmica dos sinais filtrados
## fs: MIBIF - seleciona as 8 melhores características com base na informação mútua
## clf: LDA classifier - classifica os dados

#print(results)
# Exemplo: supondo que você tenha uma função para rodar o pipeline
# results = executar_pipeline(dataset, pre_folding, pos_folding)

# Para teste, pode usar uma lista ou dicionário fictício:
results = [{'fold': 1, 'accuracy': 0.85}, {'fold': 2, 'accuracy': 0.88}]
print(results)
# Exibe uma tabela com os resultados.

import pandas as pd
#from bciflow.modules.analysis.metric_functions import accuracy

def accuracy(df):
    return df['accuracy'].mean()
# Cria uma função de acurácia simples.

df = pd.DataFrame(results)
acc = accuracy(df)

print(f"Accuracy: {acc:.4f}")
# Calcula a acurácia dos resultados obtidos e exibe o valor formatado com 4 casas decimais.

