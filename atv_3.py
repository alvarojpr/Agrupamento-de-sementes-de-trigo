import pandas as pd # Pandas é utilizada para manipulação de dados
from matplotlib import pyplot as plt # Matplotlib é utilizada para plotar gráficos
from scipy.cluster.hierarchy import linkage, dendrogram # SciPy é utilizada para agrupamento hierárquico
from sklearn.cluster import KMeans, AgglomerativeClustering # Sklearn é utilizada para algoritmos de agrupamento
from sklearn.metrics import silhouette_score # Silhouette Score é utilizado para avaliação de agrupamentos
from sklearn.preprocessing import StandardScaler # StandardScaler é utilizado para normalização de dados

# 1. Obtenha os dados de propriedades de sementes de três variedades diferentes de trigo do site
# OpenML. Os dados possuem 210 instâncias com 8 atributos cada.
dados = pd.read_csv('phpPrh7lv.csv') # Carrega o .csv em 'dados'



# 2. Pré-processamento de dados:
# - Verifique as correlações entre atributos e elimine os atributos dependente (por exemplo, use
# somente 4 atributos que tem correlação média menor).
# - Faça a normalização dos atributos independentes.
correlacoes = dados.iloc[:, :-1].corr() #faz a correlação entre os atributos sem contar a coluna de classes
print(f"",correlacoes)

cor_media = correlacoes.abs().mean().sort_values() #faz a média das correlações e ordena
print(f"",cor_media)

atributos_selecionados = cor_media[:4].index.tolist() #seleciona os 4 atributos com menor correlação
print(f"Atributos selecionados:", atributos_selecionados)

scaler = StandardScaler() # scaler para normalização
dados_normalizados = scaler.fit_transform(dados[atributos_selecionados]) # normaliza os dados dos atributos selecionados, a classe não é normalizada



# 3. Realize agrupamento k-Means (usando sklearn.cluster.KMeans) supondo 3 grupos e pontos
# centrais inicias aleatórios. Discute como melhorar a escolha dos centroids iniciais. Faça um
# agrupamento supondo 4 grupos. Calcule a coesão e separação de agrupamentos em 3 e 4 grupos e
# determine qual número de grupo é correto.
# k-Means para 4 grupos
kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(dados_normalizados) # kmeans para 4 grupos, fit é para treinar o modelo
silhouette_4 = silhouette_score(dados_normalizados, kmeans_4.labels_) # calcula o silhouette score para 4 grupos
print(f"Silhouette Score para 4 grupos: {silhouette_4}")



# 4. Realize agrupamento hierárquico (usando sklearn.cluster.AgglomerativeClustering) e produza o
# dendrograma dele. Para dicas, veja o tutorial SciPy Hierarchical Clustering and Dendrogram e
# Hierarchical Clustering. 
# Agrupamento Hierárquico e Dendrograma
modelo_hierarquico = AgglomerativeClustering(n_clusters=None, distance_threshold=0) # modelo hierárquico com número de clusters indefinido, distance_threshold é a distância máxima entre dois clusters para serem agrupados
modelo_hierarquico = modelo_hierarquico.fit(dados_normalizados)
# Geração do dendrograma
Z = linkage(dados_normalizados, method='ward') # linkage é utilizado para agrupamento hierárquico, method='ward' é um método de agrupamento hierárquico
plt.figure(figsize=(10, 7)) # define o tamanho da figura
plt.title("Dendrograma") # define o título do gráfico
dendrogram(Z) # gera o dendrograma