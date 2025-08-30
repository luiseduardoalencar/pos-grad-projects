#%%
import numpy as np

# --- Célula 1: Definição das variáveis do problema ---

# Vetor de médias (esperanças) das variáveis aleatórias X e Y
# E[X] = 1 e E[Y] = 2
mu = np.array([[1], [2]])

# Matriz de covariância de X e Y
C_XY = np.array([[4, 0],
                 [0, 9]])

print("Vetor de Médias (μ):")
print(mu)
print("\nMatriz de Covariância (C_XY):")
print(C_XY)
# %%
# --- Célula 2: Cálculo da Matriz de Correlação ---

# A matriz de correlação R_XY é calculada pela fórmula:
# R_XY = C_XY + μ * μ^T
# Onde μ * μ^T é o produto externo do vetor de médias.

# Calculando o produto externo do vetor de médias (μ * μ^T)
# O operador @ em numpy realiza a multiplicação de matrizes.
mu_mu_T = mu @ mu.T

# Calculando a matriz de correlação R_XY somando a matriz de covariância
R_XY = C_XY + mu_mu_T

print("Produto Externo das Médias (μ * μ^T):")
print(mu_mu_T)
#%%
# --- Célula 3: Apresentação do Resultado Final ---

# Exibindo a matriz de correlação R_XY teórica resultante
print("Resultado Final:")
print("A matriz de correlação teórica R_XY é:")
print(R_XY)
# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Célula 4: Definição dos Parâmetros da Distribuição ---

# Parâmetros definidos na questão anterior
# Vetor de médias (μ) para as variáveis X e Y
mu = np.array([1, 2])

# Matriz de covariância (C_XY)
# Var(X)=4, Var(Y)=9, Cov(X,Y)=0
C_XY = np.array([[4, 0],
                 [0, 9]])

# Número de amostras (pontos) a serem gerados, conforme solicitado
N = 5000

print("Parâmetros da Distribuição Gaussiana Bivariada:")
print(f"Vetor de Médias (μ): {mu}")
print(f"Matriz de Covariância (C_XY):\n{C_XY}")
print(f"Número de Amostras a gerar (N): {N}")

#%%
# --- Célula 2: Geração da Amostra S ---

# Gerando a amostra S com N=5000 pontos (x_k, y_k).
# A função np.random.multivariate_normal gera amostras de uma
# distribuição normal multivariada, dados a média e a covariância.
np.random.seed(42) # Usando uma semente para garantir que os resultados sejam reprodutíveis
S = np.random.multivariate_normal(mu, C_XY, N)

# A variável 'S' agora é uma matriz de 5000 linhas e 2 colunas,
# onde cada linha representa um par (x_k, y_k).

print(f"A amostra S foi gerada com sucesso.")
print(f"Dimensões da matriz S: {S.shape}")
print("\nExemplo das 5 primeiras amostras (x_k, y_k):")
print(S[:5].round(4))

#%%
# --- Célula 3: Verificação das Características da Amostra Gerada ---

# Para confirmar que a amostra gerada reflete as características teóricas,
# podemos calcular sua média e covariância.
# Os valores devem ser próximos aos parâmetros originais (μ e C_XY).

# Média da amostra (calculada sobre as colunas, axis=0)
mean_amostra = np.mean(S, axis=0)

# Covariância da amostra
# rowvar=False indica que as colunas representam as variáveis (X e Y)
cov_amostra = np.cov(S, rowvar=False)

print("--- Verificação Estatística da Amostra Gerada ---")
print(f"\nMédia Teórica (μ):     {mu}")
print(f"Média da Amostra S:  {mean_amostra.round(4)}")

print("\nCovariância Teórica (C_XY):")
print(C_XY)
print("\nCovariância da Amostra S:")
print(cov_amostra.round(4))

#%%
# --- Célula 4: Visualização Gráfica da Amostra S ---

# Um gráfico de dispersão (scatter plot) é ideal para visualizar os dados.

# Separando os dados de X e Y para o gráfico
x_k = S[:, 0]
y_k = S[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x_k, y_k, alpha=0.4, s=15, label=f'{N} Amostras (x_k, y_k)')
# Plotando a média teórica como referência
plt.plot(mu[0], mu[1], 'r+', markersize=15, markeredgewidth=3, label=f'Média Teórica ({mu[0]}, {mu[1]})')

plt.title(f'Amostra S com N={N} Pontos da Distribuição Gaussiana')
plt.xlabel('Variável X')
plt.ylabel('Variável Y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
# Usar 'axis('equal')' garante que a escala dos eixos seja a mesma,
# mostrando a forma correta da distribuição.
plt.axis('equal')
plt.show()
# %%
import numpy as np

# --- Célula 1: Recriação da Amostra S ---
# Para garantir a reprodutibilidade, usamos os mesmos parâmetros e semente (seed).

# Parâmetros da distribuição
mu = np.array([1, 2])
C_XY_teorica = np.array([[4, 0], [0, 9]])
N = 5000

# Usando uma semente para gerar sempre a mesma amostra "aleatória"
np.random.seed(42)
S = np.random.multivariate_normal(mu, C_XY_teorica, N)

print(f"Amostra S com {S.shape[0]} pontos foi recriada com sucesso.")

#%%
# --- Célula 2: Método 1 - Estimativa Manual (divisão por N) ---

# Esta abordagem corresponde à Estimativa de Máxima Verossimilhança (MLE).
# Var(X) = (1/N) * Σ(x_k - μ_x)²
# Cov(X,Y) = (1/N) * Σ(x_k - μ_x)(y_k - μ_y)

# 1. Calcular a média da amostra
mean_amostra = np.mean(S, axis=0)
mu_x_hat, mu_y_hat = mean_amostra

# 2. Calcular as variâncias e a covariância com divisão por N
var_x_manual = np.sum((S[:, 0] - mu_x_hat)**2) / N
var_y_manual = np.sum((S[:, 1] - mu_y_hat)**2) / N
cov_xy_manual = np.sum((S[:, 0] - mu_x_hat) * (S[:, 1] - mu_y_hat)) / N

# 3. Montar a matriz de covariância estimada manualmente
C_estimada_manual = np.array([[var_x_manual, cov_xy_manual],
                              [cov_xy_manual, var_y_manual]])

print("--- Matriz de Covariância Estimada Manualmente (Divisão por N) ---")
print(C_estimada_manual)

#%%
# --- Célula 3: Método 2 - Estimativa com numpy.cov (divisão por N-1) ---

# A função numpy.cov, por padrão, calcula a covariância amostral não-viesada.
# O divisor usado é N-1 (graus de liberdade delta, ou 'ddof', é 1 por padrão).
# rowvar=False é crucial, pois nossos dados estão em colunas (uma para X, uma para Y).

C_estimada_numpy = np.cov(S, rowvar=False)

print("--- Matriz de Covariância Estimada com numpy.cov (Divisão por N-1) ---")
print(C_estimada_numpy)
#%%
# --- Célula 4: Comparação e Análise dos Resultados ---

print("--- Comparação Final ---\n")
print("Matriz Manual (divisor N):")
print(C_estimada_manual)
print("\nMatriz com numpy.cov (divisor N-1):")
print(C_estimada_numpy)

print("\n--- Análise da Diferença ---")
print("A diferença entre as duas matrizes se deve ao fator de normalização (N vs. N-1).")
print(f"Para N = {N}, o fator é N-1 = {N-1}.")

# A relação matemática entre elas é:
# Matriz(N-1) = Matriz(N) * (N / (N-1))
# ou
# Matriz(N) = Matriz(N-1) * ((N-1) / N)

# Vamos verificar essa relação:
fator_conversao = (N - 1) / N
C_numpy_convertida = C_estimada_numpy * fator_conversao

print(f"\nConvertendo a matriz do numpy.cov (multiplicando por (N-1)/N = {fator_conversao:.5f}):")
print(C_numpy_convertida)

# A função np.allclose é usada para comparar matrizes de ponto flutuante com uma tolerância
sao_iguais = np.allclose(C_estimada_manual, C_numpy_convertida)

print(f"\nA matriz manual e a matriz do numpy convertida são idênticas? {sao_iguais}")

print("\nConclusão:")
print("A estimativa com o divisor N-1 (usada por `numpy.cov`) é chamada de 'covariância amostral não-viesada' e é o padrão na maioria dos pacotes estatísticos porque, em média, seu valor é igual à verdadeira covariância da população. A estimativa com o divisor N (manual) é o 'estimador de máxima verossimilhança', que é ligeiramente viesado (tende a subestimar a variância real). Para um N grande como 5000, a diferença entre as duas estimativas é muito pequena.")
# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Célula 1: Recriação da Amostra S ---
# Para garantir a consistência, usamos os mesmos parâmetros e semente (seed) das questões anteriores.

# Parâmetros da distribuição original
mu_original = np.array([1, 2])
C_XY_teorica = np.array([[4, 0], [0, 9]])
N = 5000

# Usando uma semente para gerar sempre a mesma amostra "aleatória"
np.random.seed(42)
S = np.random.multivariate_normal(mu_original, C_XY_teorica, N)

print(f"Amostra original S com {S.shape[0]} pontos foi recriada.")

#%%
# --- Célula 2: Estandardização de S para Gerar a Nova Amostra S1 ---

# 1. Calcular a média e o desvio padrão da amostra S para cada variável (coluna)
mean_amostra_S = np.mean(S, axis=0)
std_amostra_S = np.std(S, axis=0) # Por padrão, numpy usa o divisor N, que é o correto para estandardizar a própria amostra.

print("Estatísticas da Amostra Original S:")
print(f"Média de S: {mean_amostra_S.round(4)}")
print(f"Desvio Padrão de S: {std_amostra_S.round(4)}\n")

# 2. Aplicar a fórmula de estandardização: (dado - média) / desvio_padrão
# O numpy faz isso de forma eficiente para toda a matriz.
S1 = (S - mean_amostra_S) / std_amostra_S

print(f"Nova amostra S1 gerada com sucesso. Dimensões: {S1.shape}")
print("Exemplo das 5 primeiras amostras de S1 = {(u_k, v_k)}:")
print(S1[:5].round(4))

#%%
# --- Célula 3: Verificação das Propriedades da Nova Amostra S1 ---
# Vamos verificar se a média de S1 é de fato 0 e se a variância é 1.

mean_S1 = np.mean(S1, axis=0)
var_S1 = np.var(S1, axis=0)

print("--- Verificação das Propriedades de S1 ---")
print(f"Média da nova amostra S1 [E[U], E[V]]: {mean_S1.round(10)}")
print(f"Variância da nova amostra S1 [Var(U), Var(V)]: {var_S1.round(10)}")
print("\nComo esperado, a média é efetivamente zero e a variância é unitária.")

#%%
# --- Célula 4: Gráfico de Dispersão da Amostra Estandardizada S1 ---

# Extraindo as variáveis u e v para o plot
u_k = S1[:, 0]
v_k = S1[:, 1]

plt.figure(figsize=(9, 9))
plt.scatter(u_k, v_k, alpha=0.4, s=15)

# Adicionando elementos visuais para referência
plt.title('Gráfico de Dispersão da Amostra Estandardizada S1 = {(u_k, v_k)}')
plt.xlabel('Variável U (Estandardizada)')
plt.ylabel('Variável V (Estandardizada)')
plt.axhline(0, color='r', linestyle='--', linewidth=1, label='Média U=0')
plt.axvline(0, color='r', linestyle='--', linewidth=1, label='Média V=0')

# Usar plt.axis('equal') é fundamental para uma visualização correta da distribuição.
# Como Var(U) = Var(V) = 1, esperamos uma nuvem de pontos circular.
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
# %%
