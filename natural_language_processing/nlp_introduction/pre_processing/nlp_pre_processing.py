#%%
#NLP_PRE_PROCESSING.PY
# %%
import re
import string
import unicodedata
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, pipeline,
    BertTokenizer, BertForTokenClassification
)
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from unidecode import unidecode
# %%
# Baixar recursos do NLTK necessários
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
  nltk.download('rslp')
except:
  print("Não consegui baixar rslp")

#%%
import torch
from transformers import __version__ as transformers_version

print("=" * 50)
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🤗 Transformers: {transformers_version}")
print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 50)


# %%
@dataclass
class PreProcessingConfig:
    """Configurações para o pipeline de pré-processamento."""

    # Modelos utilizados
    ner_model: str = "pierreguillou/ner-bert-base-cased-pt-lenerbr"
    pos_model: str = "neuralmind/bert-large-portuguese-cased"
    tokenizer_model: str = "neuralmind/bert-base-portuguese-cased"

    # Configurações de limpeza
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    normalize_unicode: bool = True
    remove_extra_whitespaces: bool = True
    convert_to_lowercase: bool = True

    # Configurações de tokenização
    remove_punctuation: bool = False
    preserve_sentences: bool = True
    max_length: int = 512

    # Configurações de stopwords
    custom_stopwords: Optional[List[str]] = None
    keep_important_words: bool = True

    # Configurações de stemming/lemmatization
    use_stemming: bool = False
    use_lemmatization: bool = True

    # Configurações de dispositivo
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device configurado: {device}")  # Deve mostrar: "cuda"

# %%
class OptimizedTextPreProcessor:
    """
    Pipeline otimizado de pré-processamento de texto para português brasileiro.

    Este processador implementa todas as etapas mostradas no diagrama:
    1. Limpeza
    2. Tokenização
    3. Remoção de Stopwords
    4. POS Tagging
    5. NER Tagging
    6. Stemming & Lemmatization
    """

    def __init__(self, config: Optional[PreProcessingConfig] = None):
        """
        Inicializa o pré-processador com a configuração especificada.

        Args:
            config: Configuração personalizada. Se None, usa configuração padrão.
        """
        self.config = config or PreProcessingConfig()
        self._setup_models()
        self._setup_nltk_resources()
        self._setup_spacy()

    def _setup_models(self):
        """Inicializa os modelos transformer necessários."""
        print("🔄 Carregando modelos transformer...")

        # Tokenizer principal
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_model,
            do_lower_case=self.config.convert_to_lowercase
        )

        # Pipeline de NER
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=self.config.ner_model,
                tokenizer=self.config.ner_model,
                aggregation_strategy="simple",
                device=0 if self.config.device == "cuda" else -1
            )
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo NER: {e}")
            self.ner_pipeline = None

        # Modelo para POS tagging
        try:
            self.pos_tokenizer = AutoTokenizer.from_pretrained(self.config.pos_model)
            self.pos_model = AutoModelForTokenClassification.from_pretrained(
                self.config.pos_model
            )
            if self.config.device == "cuda":
                self.pos_model = self.pos_model.cuda()
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo POS: {e}")
            self.pos_model = None
            self.pos_tokenizer = None

        print("✅ Modelos carregados com sucesso!")

    def _setup_nltk_resources(self):
        """Configura recursos do NLTK."""
        # Stopwords em português
        self.stop_words = set(stopwords.words('portuguese'))

        # Adicionar stopwords customizadas se fornecidas
        if self.config.custom_stopwords:
            self.stop_words.update(self.config.custom_stopwords)

        # Stemmer para português
        self.stemmer = RSLPStemmer()

        # Palavras importantes que não devem ser removidas
        self.important_words = {
            'não', 'sim', 'muito', 'pouco', 'bem', 'mal', 'sempre', 'nunca',
            'todo', 'nada', 'algum', 'nenhum', 'primeiro', 'último'
        }

        if self.config.keep_important_words:
            self.stop_words -= self.important_words

    def _setup_spacy(self):
        """Configura o modelo spaCy para português."""
        try:
            # Tentar carregar modelo em português
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            print("⚠️ Modelo spaCy pt_core_news_sm não encontrado.")
            print("📥 Para instalar: python -m spacy download pt_core_news_sm")
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """
        Etapa 1: Limpeza do texto.

        Remove HTML, URLs, emails, caracteres especiais e normaliza o texto.

        Args:
            text: Texto a ser limpo

        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""

        # Remover HTML
        if self.config.remove_html:
            text = BeautifulSoup(text, "html.parser").get_text()

        # Remover URLs
        if self.config.remove_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = re.sub(url_pattern, '', text)

        # Remover emails
        if self.config.remove_emails:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            text = re.sub(email_pattern, '', text)

        # Remover números de telefone brasileiros
        if self.config.remove_phone_numbers:
            phone_patterns = [
                r'\(\d{2}\)\s*\d{4,5}-?\d{4}',  # (11) 99999-9999
                r'\d{2}\s*\d{4,5}-?\d{4}',      # 11 99999-9999
                r'\+55\s*\d{2}\s*\d{4,5}-?\d{4}' # +55 11 99999-9999
            ]
            for pattern in phone_patterns:
                text = re.sub(pattern, '', text)

        # Normalizar Unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)

        # Converter para minúsculas
        if self.config.convert_to_lowercase:
            text = text.lower()

        # Remover caracteres de controle
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

        # Remover espaços extras
        if self.config.remove_extra_whitespaces:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Etapa 2: Tokenização do texto.

        Utiliza o tokenizer do transformer para quebrar o texto em tokens.

        Args:
            text: Texto a ser tokenizado

        Returns:
            Lista de tokens
        """
        if not text:
            return []

        # Tokenização usando transformer
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Converter tokens de volta para strings
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # Filtrar tokens especiais e vazios
        tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] and token.strip()]

        # Remover pontuação se solicitado
        if self.config.remove_punctuation:
            tokens = [token for token in tokens if not all(c in string.punctuation for c in token)]

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Etapa 3: Remoção de stopwords.

        Remove palavras irrelevantes para análise, preservando palavras importantes.

        Args:
            tokens: Lista de tokens

        Returns:
            Lista de tokens sem stopwords
        """
        if not tokens:
            return []

        # Filtrar stopwords
        filtered_tokens = []
        for token in tokens:
            # Limpar token de caracteres especiais do tokenizer
            clean_token = token.replace('##', '').strip()

            if clean_token and clean_token.lower() not in self.stop_words:
                filtered_tokens.append(token)

        return filtered_tokens

    def pos_tagging(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Etapa 4: POS Tagging (Part-of-Speech).

        Identifica a classe gramatical de cada token usando modelo transformer.

        Args:
            tokens: Lista de tokens

        Returns:
            Lista de tuplas (token, pos_tag)
        """
        if not tokens or not self.pos_model:
            return [(token, 'UNKNOWN') for token in tokens]

        try:
            # Reconstruir texto dos tokens
            text = self.tokenizer.convert_tokens_to_string(tokens)

            # Tokenizar para o modelo POS
            inputs = self.pos_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )

            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Fazer predição
            with torch.no_grad():
                outputs = self.pos_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

            # Mapear predições para labels (simplificado)
            pos_labels = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'PREP', 'CONJ', 'NUM', 'PUNCT', 'OTHER']

            tagged_tokens = []
            pred_list = predictions[0].cpu().numpy()

            for i, token in enumerate(tokens):
                if i < len(pred_list):
                    pos_idx = pred_list[i]
                    if pos_idx < len(pos_labels):
                        pos_tag = pos_labels[pos_idx]
                    else:
                        pos_tag = 'OTHER'
                else:
                    pos_tag = 'OTHER'

                tagged_tokens.append((token, pos_tag))

            return tagged_tokens

        except Exception as e:
            print(f"⚠️ Erro no POS tagging: {e}")
            return [(token, 'ERROR') for token in tokens]

    def ner_tagging(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Etapa 5: NER Tagging (Named Entity Recognition).

        Identifica entidades nomeadas no texto usando modelo transformer.

        Args:
            text: Texto para análise de entidades

        Returns:
            Lista de entidades encontradas
        """
        if not text or not self.ner_pipeline:
            return []

        try:
            # Executar NER
            entities = self.ner_pipeline(text)

            # Processar e filtrar entidades
            processed_entities = []
            for entity in entities:
                if entity['score'] > 0.5:  # Filtrar por confiança
                    processed_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0)
                    })

            return processed_entities

        except Exception as e:
            print(f"⚠️ Erro no NER tagging: {e}")
            return []

    def stem_and_lemmatize(self, tokens: List[str]) -> Dict[str, List[str]]:
        """
        Etapa 6: Stemming e Lemmatização.

        Reduz palavras às suas formas radicais (stemming) e canônicas (lemmatização).

        Args:
            tokens: Lista de tokens

        Returns:
            Dicionário com tokens originais, stemmed e lemmatized
        """
        if not tokens:
            return {'original': [], 'stemmed': [], 'lemmatized': []}

        stemmed_tokens = []
        lemmatized_tokens = []

        for token in tokens:
            # Remover prefixos do tokenizer
            clean_token = token.replace('##', '').strip()

            if not clean_token:
                continue

            # Stemming
            if self.config.use_stemming:
                stemmed = self.stemmer.stem(clean_token.lower())
                stemmed_tokens.append(stemmed)
            else:
                stemmed_tokens.append(clean_token)

            # Lemmatização usando spaCy
            if self.config.use_lemmatization and self.nlp:
                doc = self.nlp(clean_token)
                if doc:
                    lemmatized = doc[0].lemma_.lower()
                    lemmatized_tokens.append(lemmatized)
                else:
                    lemmatized_tokens.append(clean_token.lower())
            else:
                lemmatized_tokens.append(clean_token.lower())

        return {
            'original': tokens,
            'stemmed': stemmed_tokens,
            'lemmatized': lemmatized_tokens
        }

    def process_pipeline(self, text: str, return_intermediate: bool = False) -> Dict:
        """
        Executa o pipeline completo de pré-processamento.

        Args:
            text: Texto a ser processado
            return_intermediate: Se True, retorna resultados intermediários

        Returns:
            Dicionário com resultados do processamento
        """
        results = {'original_text': text}

        # Etapa 1: Limpeza
        cleaned_text = self.clean_text(text)
        if return_intermediate:
            results['cleaned_text'] = cleaned_text

        # Etapa 2: Tokenização
        tokens = self.tokenize_text(cleaned_text)
        if return_intermediate:
            results['tokens'] = tokens

        # Etapa 3: Remoção de stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        if return_intermediate:
            results['filtered_tokens'] = filtered_tokens

        # Etapa 4: POS Tagging
        pos_tagged = self.pos_tagging(filtered_tokens)
        results['pos_tags'] = pos_tagged

        # Etapa 5: NER Tagging
        entities = self.ner_tagging(cleaned_text)
        results['entities'] = entities

        # Etapa 6: Stemming e Lemmatização
        stem_lemma_results = self.stem_and_lemmatize(filtered_tokens)
        results.update(stem_lemma_results)

        return results

    def batch_process(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Processa uma lista de textos em lotes para maior eficiência.

        Args:
            texts: Lista de textos para processar
            batch_size: Tamanho do lote

        Returns:
            Lista de resultados do processamento
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []

            for text in batch:
                result = self.process_pipeline(text)
                batch_results.append(result)

            results.extend(batch_results)
            print(f"✅ Processados {min(i + batch_size, len(texts))}/{len(texts)} textos")

        return results
# %%
texto_exemplo = """
    <p>Olá! Este é um exemplo de texto em português brasileiro para demonstrar
    o pipeline de pré-processamento. Visitei o site https://www.somosicev.com.br
    e encontrei informações sobre a Faculdade iCEV.</p>

    A inteligência artificial está revolucionando a análise de texto!
    """

#%%
# Criar configuração personalizada
config = PreProcessingConfig(
    convert_to_lowercase=True,
    remove_html=True,
    use_lemmatization=True,
    keep_important_words=True
)

# Processar texto
processor = OptimizedTextPreProcessor(config)
# %%
resultado = processor.process_pipeline(texto_exemplo, return_intermediate=True)
print("🔤 TEXTO ORIGINAL:")
print(resultado['original_text'])
print("\n🧹 TEXTO LIMPO:")
print(resultado['cleaned_text'])
print("\n🔧 TOKENS:")
print(resultado['tokens'][:15])  # Primeiros 10 tokens
print("\n🚫 TOKENS SEM STOPWORDS:")
print(resultado['filtered_tokens'][:10])
print("\n🏷️ POS TAGS:")
print(resultado['pos_tags'][:10])
print("\n🎯 ENTIDADES:")
print(resultado['entities'])
print("\n🌿 TOKENS LEMMATIZADOS:")
print(resultado['lemmatized'][:15])
# %%


#%%

