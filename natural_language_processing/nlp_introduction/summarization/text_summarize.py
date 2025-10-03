#%%
#TEXT_SUMMARIZE.PY
#%%
import pandas as pd
import numpy as np
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings
from typing import List, Dict, Optional
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")
# %%
# Configurações globais
MODELO_SUMARIZACAO = "google-t5/t5-small"  # T5 Small do Google
MODELO_ALTERNATIVO = "pierreguillou/t5-small-pt-summarize"   # T5 Small PT
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 150
MIN_OUTPUT_LENGTH = 30
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class PortugueseSummarizer:
    """
    Sumarizador de texto para português brasileiro usando T5.
    """

    def __init__(self, model_name: str = MODELO_SUMARIZACAO, use_pipeline: bool = True):
        """
        Inicializa o sumarizador.

        Args:
            model_name: Nome do modelo pré-treinado
            use_pipeline: Se deve usar pipeline do transformers (mais simples)
        """
        self.model_name = model_name
        self.device = DEVICE
        self.use_pipeline = use_pipeline

        print(f"🔧 Inicializando sumarizador de texto PT-BR...")
        print(f"   • Modelo: {model_name}")
        print(f"   • Dispositivo: {self.device}")
        print(f"   • Modo: {'Pipeline' if use_pipeline else 'Manual'}")

        self._setup_model()
        print("✅ Sumarizador inicializado com sucesso!")

    def _setup_model(self):
        """Configura o modelo e tokenizer."""
        try:
            if self.use_pipeline:
                # Usar pipeline para simplificar
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0 if self.device.type == 'cuda' else -1
                )
                self.tokenizer = None
                self.model = None
            else:
                # Configuração manual para mais controle
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.summarizer = None

        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo principal: {e}")
            print("🔄 Tentando modelo alternativo...")

            # Fallback para modelo alternativo
            try:
                if self.use_pipeline:
                    self.summarizer = pipeline(
                        "summarization",
                        model=MODELO_ALTERNATIVO,
                        device=0 if self.device.type == 'cuda' else -1
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(MODELO_ALTERNATIVO)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(MODELO_ALTERNATIVO)
                    self.model.to(self.device)

                print(f"✅ Modelo alternativo carregado: {MODELO_ALTERNATIVO}")

            except Exception as e2:
                print(f"❌ Erro com modelo alternativo: {e2}")
                raise Exception("Não foi possível carregar nenhum modelo de sumarização")

    def summarize_text(self, text: str, max_length: int = MAX_OUTPUT_LENGTH,
                      min_length: int = MIN_OUTPUT_LENGTH) -> str:
        """
        Sumariza um texto.

        Args:
            text: Texto a ser sumarizado
            max_length: Comprimento máximo do resumo
            min_length: Comprimento mínimo do resumo

        Returns:
            Texto sumarizado
        """
        if not text or len(text.strip()) < 50:
            return "Texto muito curto para sumarização."

        # Limpar e preparar texto
        cleaned_text = self._clean_text(text)

        # Truncar se necessário
        if len(cleaned_text.split()) > MAX_INPUT_LENGTH:
            words = cleaned_text.split()[:MAX_INPUT_LENGTH]
            cleaned_text = ' '.join(words)

        try:
            if self.use_pipeline:
                # Usar pipeline
                result = self.summarizer(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4
                )
                return result[0]['summary_text']

            else:
                # Método manual
                inputs = self.tokenizer(
                    cleaned_text,
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )

                summary = self.tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                )
                return summary

        except Exception as e:
            print(f"⚠️ Erro na sumarização: {e}")
            return f"Erro ao sumarizar o texto: {str(e)}"

    def _clean_text(self, text: str) -> str:
        """Limpa e prepara o texto para sumarização."""
        # Remover quebras de linha excessivas
        text = re.sub(r'\n+', ' ', text)

        # Remover espaços excessivos
        text = re.sub(r'\s+', ' ', text)

        # Remover caracteres especiais desnecessários
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)

        return text.strip()

    def batch_summarize(self, texts: List[str], max_length: int = MAX_OUTPUT_LENGTH) -> List[str]:
        """
        Sumariza múltiplos textos.

        Args:
            texts: Lista de textos para sumarizar
            max_length: Comprimento máximo dos resumos

        Returns:
            Lista de textos sumarizados
        """
        summaries = []

        print(f"📝 Sumarizando {len(texts)} textos...")

        for i, text in enumerate(tqdm(texts, desc="Sumarizando")):
            try:
                summary = self.summarize_text(text, max_length=max_length)
                summaries.append(summary)

                # Pequena pausa para evitar sobrecarga
                if i % 5 == 0:
                    time.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Erro no texto {i+1}: {e}")
                summaries.append("Erro na sumarização")

        return summaries

    def analyze_summaries(self, original_texts: List[str], summaries: List[str]) -> Dict:
        """
        Analisa as sumarizações criadas.

        Args:
            original_texts: Textos originais
            summaries: Resumos criados

        Returns:
            Dicionário com estatísticas
        """
        if len(original_texts) != len(summaries):
            raise ValueError("Número de textos originais e resumos deve ser igual")

        stats = {
            'num_texts': len(original_texts),
            'original_lengths': [len(text.split()) for text in original_texts],
            'summary_lengths': [len(summary.split()) for summary in summaries],
            'compression_ratios': []
        }

        # Calcular taxas de compressão
        for orig, summ in zip(stats['original_lengths'], stats['summary_lengths']):
            if orig > 0:
                ratio = summ / orig
                stats['compression_ratios'].append(ratio)
            else:
                stats['compression_ratios'].append(0)

        # Estatísticas agregadas
        stats['avg_original_length'] = np.mean(stats['original_lengths'])
        stats['avg_summary_length'] = np.mean(stats['summary_lengths'])
        stats['avg_compression_ratio'] = np.mean(stats['compression_ratios'])

        return stats
# %%
dataset = load_dataset(
      "wikimedia/wikipedia",
      "20231101.pt",
      split="train",
      streaming=True
  )
articles = []
count = 0
num_articles = 3
# %%
summarizer = PortugueseSummarizer(use_pipeline=True)
# %%
texto_exemplo = """
    A inteligência artificial é uma das tecnologias mais promissoras do século XXI.
    Ela tem potencial para revolucionar diversos setores, desde a medicina até a educação.
    No Brasil, várias universidades e empresas estão investindo pesadamente em pesquisa
    e desenvolvimento de IA. As aplicações incluem reconhecimento de voz, processamento
    de linguagem natural, visão computacional e aprendizado de máquina. Apesar dos
    benefícios, também existem desafios éticos e sociais que precisam ser considerados,
    como privacidade de dados, substituição de empregos e viés algorítmico. O futuro
    da IA no país dependerá de políticas públicas adequadas e investimento contínuo
    em educação e pesquisa.
    """
resumo_teste = summarizer.summarize_text(texto_exemplo)
print(f"📄 Texto original: {len(texto_exemplo.split())} palavras")
print(f"📋 Resumo: {resumo_teste}")
print(f"📊 Resumo: {len(resumo_teste.split())} palavras")
# %%
for article in dataset.shuffle(seed=42):
  if count >= num_articles:
      break
  # Filtrar artigos muito pequenos ou muito grandes
  text = article['text']
  if text and 200 <= len(text.split()) <= 1000:
      articles.append({
          'title': article['title'],
          'text': text,
          'url': article.get('url', ''),
          'word_count': len(text.split())
      })
      count += 1


print(f"✅ {len(articles)} artigos carregados com sucesso!")
# %%
articles[0]
# %%
results = []
max_summary_length = 120
for i, article in enumerate(tqdm(articles, desc="Sumarizando artigos")):
  # Sumarizar o artigo
  summary = summarizer.summarize_text(
      article['text'],
      max_length=max_summary_length
  )
  result = {
            'title': article['title'],
            'original_text': article['text'],
            'summary': summary,
            'original_word_count': article['word_count'],
            'summary_word_count': len(summary.split()),
            'compression_ratio': len(summary.split()) / article['word_count']
        }

  results.append(result)
  if (i + 1) % 2 == 0:
    print(f"   ✅ Processados {i+1}/{len(articles)} artigos")

# %%
for i, result in enumerate(results):
  orig_words = result['original_text'].split()[:50]
  orig_preview = ' '.join(orig_words) + "..."
  print(f"📝 TEXTO ORIGINAL ({result['original_word_count']} palavras):")
  print(orig_preview)

  print(f"\n📋 RESUMO ({result['summary_word_count']} palavras):")
  print(result['summary'])

  print(f"\n📊 COMPRESSÃO: {result['compression_ratio']:.2%}")
  print("-" * 50)
# %%
