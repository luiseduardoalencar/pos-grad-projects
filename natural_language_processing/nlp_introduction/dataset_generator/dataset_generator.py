#%%
#DATASET_GENERATOR.PY
#%%
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pre_processing import OptimizedTextPreProcessor, PreProcessingConfig
from datasets import load_dataset

print("Cargando dataset da Wikipedia...")

ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train")

# %%

config = PreProcessingConfig(
    convert_to_lowercase=True,  
    remove_html=True,
    use_lemmatization=True,
    keep_important_words=True
)

processor = OptimizedTextPreProcessor(config)
print("Processador criado com sucesso!")

#%%
ds.select(range(1))
ds[0]
# %%
artigos_exemplo = ds.shuffle(seed=42).select(range(4))
resultados = processor.batch_process(
        [artigo['text'][:1000] for artigo in artigos_exemplo],  # Primeiros 1000 chars de cada artigo
        batch_size=2
    )

# %%
for resultado in resultados:
  print(f"   ✅ Tokens processados: {len(resultado.get('lemmatized', []))}")
  print(f"   🎯 Entidades encontradas: {len(resultado.get('entities', []))}")
  print("*"*100)
# %%
emotions = load_dataset("emotion")
# %%
emotions 
# %%
emotions['train'][0]
# %%
emotions['train'].features
# %%

#Criando um dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import os
from typing import List, Tuple, Dict
import random

#%%
def gerar_dados_exemplo() -> List[Dict[str, str]]:
    """
    Gera dados de exemplo com textos em português e labels de categorias.

    Returns:
        List[Dict]: Lista com dicionários contendo 'text' e 'label'
    """

    # Dados de exemplo com diferentes categorias
    dados_exemplo = [
        # Categoria: Tecnologia
        {"text": "A inteligência artificial está revolucionando o mercado de trabalho brasileiro.", "label": "tecnologia"},
        {"text": "Smartphones com 5G chegam ao Brasil com preços mais acessíveis.", "label": "tecnologia"},
        {"text": "Aplicativo de delivery ganha nova funcionalidade de rastreamento em tempo real.", "label": "tecnologia"},
        {"text": "Empresa brasileira desenvolve software de reconhecimento facial.", "label": "tecnologia"},
        {"text": "Startup do Rio cria plataforma de educação online com IA.", "label": "tecnologia"},

        # Categoria: Esportes
        {"text": "Seleção brasileira se prepara para as próximas eliminatórias da Copa do Mundo.", "label": "esportes"},
        {"text": "Flamengo vence clássico no Maracanã com gol nos últimos minutos.", "label": "esportes"},
        {"text": "Atleta brasileiro conquista medalha de ouro nas Olimpíadas de Paris.", "label": "esportes"},
        {"text": "Campeonato Brasileiro tem nova rodada com jogos decisivos para o título.", "label": "esportes"},
        {"text": "Técnico da seleção anuncia convocação para os próximos jogos.", "label": "esportes"},

        # Categoria: Política
        {"text": "Congresso Nacional aprova nova lei de proteção de dados pessoais.", "label": "politica"},
        {"text": "Presidente anuncia investimentos em infraestrutura para o Nordeste.", "label": "politica"},
        {"text": "Ministério da Saúde lança campanha nacional de vacinação.", "label": "politica"},
        {"text": "Senado discute projeto de lei sobre energias renováveis.", "label": "politica"},
        {"text": "Governadores se reúnem para discutir políticas públicas de educação.", "label": "politica"},

        # Categoria: Economia
        {"text": "Inflação no Brasil mostra sinais de desaceleração no último trimestre.", "label": "economia"},
        {"text": "Dólar fecha em alta após decisão do Banco Central sobre taxa de juros.", "label": "economia"},
        {"text": "Setor agrícola brasileiro bate recorde de exportações este ano.", "label": "economia"},
        {"text": "Empresas de tecnologia lideram criação de empregos no país.", "label": "economia"},
        {"text": "Bolsa de valores registra maior alta do mês com otimismo dos investidores.", "label": "economia"},

        # Categoria: Saúde
        {"text": "Novo tratamento para diabetes tipo 2 mostra resultados promissores.", "label": "saude"},
        {"text": "Hospital em São Paulo realiza primeiro transplante de coração do ano.", "label": "saude"},
        {"text": "Pesquisa brasileira identifica nova proteína relacionada ao Alzheimer.", "label": "saude"},
        {"text": "Campanha de prevenção ao câncer de mama alcança milhões de mulheres.", "label": "saude"},
        {"text": "SUS amplia cobertura de medicamentos para doenças raras.", "label": "saude"},

        # Categoria: Educação
        {"text": "Universidade pública lança curso de inteligência artificial gratuito.", "label": "educacao"},
        {"text": "MEC anuncia investimento em laboratórios de ciências para escolas públicas.", "label": "educacao"},
        {"text": "Programa de bolsas estudantis beneficia milhares de jovens brasileiros.", "label": "educacao"},
        {"text": "Ensino híbrido ganha espaço nas instituições de ensino superior.", "label": "educacao"},
        {"text": "Projeto de lei propõe aumento de vagas em cursos técnicos profissionalizantes.", "label": "educacao"},

        # Categoria: Cultura
        {"text": "Festival de cinema brasileiro premia documentário sobre Amazônia.", "label": "cultura"},
        {"text": "Museu Nacional reabre com nova exposição sobre história indígena.", "label": "cultura"},
        {"text": "Livro de autor brasileiro ganha prêmio literário internacional.", "label": "cultura"},
        {"text": "Banda de rock nacional faz turnê pelos principais festivais do país.", "label": "cultura"},
        {"text": "Teatro Municipal apresenta nova temporada de ópera clássica.", "label": "cultura"},

        # Categoria: Meio Ambiente
        {"text": "Projeto de reflorestamento na Mata Atlântica planta milhão de árvores.", "label": "meio_ambiente"},
        {"text": "Energia solar cresce 50% no Brasil e bate novo recorde de geração.", "label": "meio_ambiente"},
        {"text": "ONGs denunciam aumento do desmatamento na região amazônica.", "label": "meio_ambiente"},
        {"text": "Cidade brasileira implementa sistema de reciclagem 100% sustentável.", "label": "meio_ambiente"},
        {"text": "Pesquisadores descobrem nova espécie de peixe em rio do Pantanal.", "label": "meio_ambiente"},
    ]

    return dados_exemplo
# %%
def expandir_dados_com_variacoes(dados_base: List[Dict[str, str]], fator_multiplicacao: int = 3) -> List[Dict[str, str]]:
    """
    Expande os dados base criando variações para ter mais exemplos.

    Args:
        dados_base: Dados originais
        fator_multiplicacao: Quantas vezes multiplicar os dados

    Returns:
        List[Dict]: Dados expandidos com variações
    """

    dados_expandidos = dados_base.copy()

    # Palavras para criar variações
    sinonimos = {
        "brasileiro": ["nacional", "do Brasil", "tupiniquim"],
        "nova": ["recente", "moderna", "atual"],
        "grande": ["enorme", "gigante", "massiva"],
        "pequeno": ["reduzido", "mínimo", "compacto"],
        "importante": ["relevante", "significativo", "crucial"],
        "melhor": ["superior", "excelente", "ótimo"],
        "aumenta": ["cresce", "eleva", "amplia"],
        "diminui": ["reduz", "baixa", "cai"]
    }

    for _ in range(fator_multiplicacao):
        for item in dados_base:
            texto_variado = item["text"]

            # Aplicar algumas variações aleatórias
            for palavra_original, substitutos in sinonimos.items():
                if palavra_original in texto_variado.lower():
                    if random.random() < 0.3:  # 30% de chance de substituir
                        substituto = random.choice(substitutos)
                        texto_variado = texto_variado.replace(palavra_original, substituto)

            # Adicionar variações de pontuação
            if random.random() < 0.2:
                texto_variado = texto_variado.replace(".", "!")

            dados_expandidos.append({
                "text": texto_variado,
                "label": item["label"]
            })

    return dados_expandidos
# %%
def criar_arquivo_csv(caminho_arquivo: str = "dataset_exemplo.csv", expandir: bool = True) -> str:
    """
    Cria um arquivo CSV com dados de exemplo.

    Args:
        caminho_arquivo: Caminho onde salvar o CSV
        expandir: Se deve expandir os dados com variações

    Returns:
        str: Caminho do arquivo criado
    """

    print("📝 CRIANDO ARQUIVO CSV COM DADOS DE EXEMPLO")
    print("=" * 50)

    # Gerar dados base
    dados = gerar_dados_exemplo()
    print(f"📊 Dados base gerados: {len(dados)} exemplos")

    # Expandir dados se solicitado
    if expandir:
        print("🔄 Expandindo dados com variações...")
        dados = expandir_dados_com_variacoes(dados, fator_multiplicacao=2)
        print(f"📈 Dados expandidos: {len(dados)} exemplos")

    # Embaralhar os dados
    random.shuffle(dados)

    # Converter para DataFrame
    df = pd.DataFrame(dados)

    # Mostrar estatísticas
    print(f"\n📊 ESTATÍSTICAS DO DATASET:")
    print(f"  • Total de exemplos: {len(df)}")
    print(f"  • Colunas: {list(df.columns)}")
    print(f"  • Labels únicas: {df['label'].nunique()}")

    print(f"\n🏷️ DISTRIBUIÇÃO POR LABEL:")
    distribuicao = df['label'].value_counts()
    for label, count in distribuicao.items():
        print(f"  • {label}: {count} exemplos")

    # Salvar CSV
    caminho_completo = os.path.abspath(caminho_arquivo)
    df.to_csv(caminho_completo, index=False, encoding='utf-8')

    print(f"\n✅ CSV salvo em: {caminho_completo}")
    print(f"📏 Tamanho do arquivo: {os.path.getsize(caminho_completo) / 1024:.1f} KB")

    # Mostrar preview
    print(f"\n👀 PREVIEW DOS PRIMEIROS 3 EXEMPLOS:")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. [{row['label']}] {row['text'][:80]}...")

    return caminho_completo
# %%
def ler_csv_e_criar_dataset(caminho_csv: str, test_size: float = 0.2, random_state: int = 42) -> DatasetDict:
    """
    Lê o arquivo CSV e cria um dataset com divisões train/test.

    Args:
        caminho_csv: Caminho para o arquivo CSV
        test_size: Proporção para teste (0.2 = 20%)
        random_state: Seed para reprodutibilidade

    Returns:
        DatasetDict: Dataset com divisões 'train' e 'test'
    """

    print(f"\n📖 LENDO CSV E CRIANDO DATASET")
    print("=" * 40)

    # Ler CSV
    print(f"📂 Carregando arquivo: {caminho_csv}")
    df = pd.read_csv(caminho_csv, encoding='utf-8')

    print(f"✅ Arquivo carregado:")
    print(f"  • Linhas: {len(df)}")
    print(f"  • Colunas: {list(df.columns)}")

    # Verificar dados
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV deve ter colunas 'text' e 'label'")

    # Remover linhas vazias
    df_limpo = df.dropna(subset=['text', 'label']).copy()
    print(f"🧹 Após limpeza: {len(df_limpo)} linhas")

    # Dividir em train/test
    print(f"✂️ Dividindo dataset (train: {1-test_size:.0%}, test: {test_size:.0%})")

    train_df, test_df = train_test_split(
        df_limpo,
        test_size=test_size,
        random_state=random_state,
        stratify=df_limpo['label']  # Manter proporção de labels
    )

    print(f"📊 Divisão realizada:")
    print(f"  • Train: {len(train_df)} exemplos")
    print(f"  • Test: {len(test_df)} exemplos")

    # Converter para Dataset do HuggingFace
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    })

    print(f"\n🎯 DATASET CRIADO COM SUCESSO!")
    print(f"  • Tipo: {type(dataset_dict)}")
    print(f"  • Divisões: {list(dataset_dict.keys())}")

    # Mostrar distribuição por divisão
    for split_name, split_dataset in dataset_dict.items():
        print(f"\n📈 Distribuição {split_name.upper()}:")
        labels = [item['label'] for item in split_dataset]
        distribuicao = pd.Series(labels).value_counts()
        for label, count in distribuicao.items():
            print(f"  • {label}: {count}")

    return dataset_dict
# %%
# Passo 1: Criar CSV
print("\n📋 PASSO 1: CRIANDO ARQUIVO CSV")
caminho_csv = criar_arquivo_csv("meu_dataset.csv", expandir=True)

# %%
print("\n📖 PASSO 2: CRIANDO DATASET A PARTIR DO CSV")
dataset = ler_csv_e_criar_dataset(caminho_csv, test_size=0.25)
# %%
dataset 
# %%
dataset.set_format(type="pandas")
df = dataset["train"][:]
df.head()
# %%
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from wordcloud import WordCloud


# Configurar estilo dos gráficos
try:
    plt.style.use('seaborn-v0_8')
except:
    # Se o estilo seaborn-v0_8 não estiver disponível, usar um alternativo
    plt.style.use('default')

sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Configurar matplotlib para usar fontes disponíveis no Linux
# Verificar fontes disponíveis e usar apenas as que existem
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]

# Lista de fontes preferidas (em ordem de preferência)
preferred_fonts = ['DejaVu Sans', 'Liberation Sans', 'Ubuntu', 'Cantarell', 'sans-serif']
font_family = []

for font in preferred_fonts:
    if font in available_fonts or font == 'sans-serif':
        font_family.append(font)

plt.rcParams['font.family'] = font_family if font_family else ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Suprimir warning específico de fontes
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# %%
def plotar_distribuicao(df: pd.DataFrame):
  if 'label' not in df.columns:
    print("❌ Coluna 'label' não encontrada!")
    return
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
  label_counts = df['label'].value_counts()
  bars = ax1.bar(label_counts.index, label_counts.values,
                  color=sns.color_palette("husl", len(label_counts)))
  ax1.set_title('📊 Distribuição de Classes (Labels)', fontsize=14, fontweight='bold')
  ax1.set_xlabel('Classes', fontsize=12)
  ax1.set_ylabel('Número de Registros', fontsize=12)
  ax1.tick_params(axis='x', rotation=45)
  # Adicionar valores nas barras
  for bar in bars:
      height = bar.get_height()
      ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
              f'{int(height)}', ha='center', va='bottom', fontweight='bold')

  # Gráfico de pizza
  colors = sns.color_palette("husl", len(label_counts))
  wedges, texts, autotexts = ax2.pie(label_counts.values, labels=label_counts.index,
                                    autopct='%1.1f%%', colors=colors, startangle=90)
  ax2.set_title('🥧 Proporção de Classes', fontsize=14, fontweight='bold')

  # Melhorar aparência do gráfico de pizza
  for autotext in autotexts:
      autotext.set_color('white')
      autotext.set_fontweight('bold')
      autotext.set_fontsize(10)

  plt.tight_layout()

  plt.show()

plotar_distribuicao(df)

# %%
def plotar_distribuicao_comprimento_texto(df: pd.DataFrame, salvar_fig: bool = True):
    """
    Plota a distribuição do comprimento dos textos.

    Args:
        df: DataFrame do pandas
        salvar_fig: Se deve salvar a figura
    """
    if 'text' not in df.columns:
        print("❌ Coluna 'text' não encontrada!")
        return

    # Calcular comprimentos
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()

    # Configurar figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Histograma de comprimento de caracteres
    ax1.hist(df['text_length'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('📏 Distribuição: Comprimento dos Textos (Caracteres)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Número de Caracteres', fontsize=10)
    ax1.set_ylabel('Frequência', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Adicionar linha da média
    media_chars = df['text_length'].mean()
    ax1.axvline(media_chars, color='red', linestyle='--', linewidth=2,
                label=f'Média: {media_chars:.1f}')
    ax1.legend()

    # Histograma de número de palavras
    ax2.hist(df['word_count'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('📝 Distribuição: Número de Palavras', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Número de Palavras', fontsize=10)
    ax2.set_ylabel('Frequência', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Adicionar linha da média
    media_words = df['word_count'].mean()
    ax2.axvline(media_words, color='red', linestyle='--', linewidth=2,
                label=f'Média: {media_words:.1f}')
    ax2.legend()

    # Box plot por classe (comprimento)
    if 'label' in df.columns:
        sns.boxplot(data=df, x='label', y='text_length', ax=ax3)
        ax3.set_title('📦 Box Plot: Comprimento por Classe', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Classe', fontsize=10)
        ax3.set_ylabel('Número de Caracteres', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)

        # Violin plot por classe (palavras)
        sns.violinplot(data=df, x='label', y='word_count', ax=ax4)
        ax4.set_title('🎻 Violin Plot: Palavras por Classe', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Classe', fontsize=10)
        ax4.set_ylabel('Número de Palavras', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if salvar_fig:
        plt.savefig('distribuicao_comprimento.png', dpi=300, bbox_inches='tight')
        print("💾 Gráfico salvo como 'distribuicao_comprimento.png'")

    plt.show()

def criar_nuvem_palavras(df: pd.DataFrame, salvar_fig: bool = True):
    """
    Cria nuvens de palavras para cada classe.

    Args:
        df: DataFrame do pandas
        salvar_fig: Se deve salvar a figura
    """
    if 'text' not in df.columns or 'label' not in df.columns:
        print("❌ Colunas 'text' ou 'label' não encontradas!")
        return

    # Obter classes únicas
    classes = df['label'].unique()
    n_classes = len(classes)

    # Configurar figura
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for i, classe in enumerate(classes):
        # Filtrar textos da classe
        textos_classe = df[df['label'] == classe]['text'].str.cat(sep=' ')

        # Limpar texto básico
        textos_classe = re.sub(r'[^\w\s]', ' ', textos_classe.lower())
        textos_classe = re.sub(r'\s+', ' ', textos_classe)

        # Criar nuvem de palavras
        try:
            wordcloud = WordCloud(
                width=800, height=600,
                background_color='white',
                max_words=50,
                colormap='viridis',
                font_path=None,
                relative_scaling=0.5,
                random_state=42
            ).generate(textos_classe)

            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'☁️ Nuvem de Palavras: {classe.upper()}',
                             fontsize=14, fontweight='bold')
            axes[i].axis('off')

        except Exception as e:
            print(f"⚠️ Erro ao criar nuvem para {classe}: {e}")
            axes[i].text(0.5, 0.5, f'Erro ao gerar\nnuvem para {classe}',
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12)

    plt.tight_layout()

    if salvar_fig:
        plt.savefig('nuvens_palavras.png', dpi=300, bbox_inches='tight')
        print("💾 Gráfico salvo como 'nuvens_palavras.png'")

    plt.show()

def analise_palavras_frequentes(df: pd.DataFrame, top_n: int = 15):
    """
    Analisa e plota as palavras mais frequentes.

    Args:
        df: DataFrame do pandas
        top_n: Número de palavras mais frequentes para mostrar
    """
    if 'text' not in df.columns:
        print("❌ Coluna 'text' não encontrada!")
        return

    print(f"\n📊 ANÁLISE DAS {top_n} PALAVRAS MAIS FREQUENTES")
    print("="*50)

    # Extrair todas as palavras
    todas_palavras = []
    for texto in df['text']:
        # Limpeza básica
        texto_limpo = re.sub(r'[^\w\s]', ' ', texto.lower())
        palavras = texto_limpo.split()
        todas_palavras.extend(palavras)

    # Contar frequências
    contador_palavras = Counter(todas_palavras)
    palavras_freq = contador_palavras.most_common(top_n)

    # Mostrar no console
    print("\n🔤 Palavras mais frequentes:")
    for i, (palavra, freq) in enumerate(palavras_freq, 1):
        print(f"   {i:2d}. {palavra:<15} → {freq:3d} ocorrências")

    # Plotar gráfico
    palavras, frequencias = zip(*palavras_freq)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(palavras)), frequencias,
                   color=sns.color_palette("viridis", len(palavras)))

    plt.title(f'📊 Top {top_n} Palavras Mais Frequentes', fontsize=16, fontweight='bold')
    plt.xlabel('Palavras', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.xticks(range(len(palavras)), palavras, rotation=45, ha='right')

    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Salvar
    plt.savefig('palavras_frequentes.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico salvo como 'palavras_frequentes.png'")

    plt.show()
# %%
def criar_dashboard_completo(df: pd.DataFrame):
    """
    Cria um dashboard completo com todas as visualizações.

    Args:
        df: DataFrame do pandas
    """
    print("\n" + "="*60)
    print("📊 CRIANDO DASHBOARD COMPLETO DE VISUALIZAÇÕES")
    print("="*60)

    # Gráfico 1: Distribuição de labels
    print("\n🎯 Criando gráfico de distribuição de labels...")
    plotar_distribuicao(df)

    # Gráfico 2: Distribuição de comprimento
    print("\n📏 Criando gráfico de distribuição de comprimento...")
    plotar_distribuicao_comprimento_texto(df)

    # Gráfico 3: Palavras frequentes
    print("\n🔤 Analisando palavras mais frequentes...")
    analise_palavras_frequentes(df)

    # Gráfico 4: Nuvens de palavras
    print("\n☁️ Criando nuvens de palavras por classe...")
    try:
        criar_nuvem_palavras(df)
    except Exception as e:
        print(f"⚠️ Erro ao criar nuvens de palavras: {e}")
        print("💡 Instale wordcloud: pip install wordcloud")

    print("\n✅ DASHBOARD COMPLETO CRIADO!")
    print("📁 Arquivos salvos:")
    print("   • distribuicao_labels.png")
    print("   • distribuicao_comprimento.png")
    print("   • palavras_frequentes.png")
    print("   • nuvens_palavras.png")
#%%
criar_dashboard_completo(df)

# %%
