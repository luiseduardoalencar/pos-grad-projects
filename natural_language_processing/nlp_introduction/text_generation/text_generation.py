#%%
#TEXT_GENERATION.PY
#%%

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from summarization.text_summarize import PortugueseSummarizer
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
    set_seed
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import time
import random

warnings.filterwarnings("ignore")
# %%
MODELO_PRINCIPAL = "bartowski/granite-embedding-107m-multilingual-GGUF"
MAX_LENGTH = 200
MIN_LENGTH = 50
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
NUM_BEAMS = 4
REPETITION_PENALTY = 1.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class TextGenerator:
    def __init__(self, model_name: str = MODELO_PRINCIPAL, use_pipeline: bool = True):
        """
        Inicializa o gerador de texto.

        Args:
            model_name: Nome do modelo pré-treinado
            use_pipeline: Se deve usar pipeline do transformers
        """
        self.model_name = model_name
        self.device = DEVICE
        self.use_pipeline = use_pipeline

        print(f"🤖 Inicializando gerador de texto PT-BR...")
        print(f"   • Modelo: {model_name}")
        print(f"   • Dispositivo: {self.device}")
        print(f"   • Modo: {'Pipeline' if use_pipeline else 'Manual'}")

        self._setup_model()
        print("✅ Gerador inicializado com sucesso!")

    def _setup_model(self):
        """Configura o modelo e tokenizer."""
        model_name = self.model_name

        try:
            print(f"🔄 Carregando modelo Granite Multilingual: {model_name}")

            if self.use_pipeline:
                # Usar pipeline para geração de texto - configuração otimizada para GGUF
                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device.type == 'cuda' else -1,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    trust_remote_code=True,
                    model_kwargs={
                        "torch_dtype": torch.float16 if self.device.type == 'cuda' else torch.float32,
                        "low_cpu_mem_usage": True,
                        "device_map": "auto" if self.device.type == 'cuda' else None
                    }
                )
                self.tokenizer = None
                self.model = None
            else:
                # Configuração manual para mais controle com GGUF
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device.type == 'cuda' else None
                )
                if self.device.type != 'cuda':
                    self.model.to(self.device)
                self.generator = None

                # Configurar pad_token se não existir
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Modelo Granite Multilingual carregado com sucesso!")
            return

        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo Granite: {e}")
            print("💡 Criando gerador simples baseado em templates...")
            self._setup_simple_generator()

    def _setup_simple_generator(self):
        """Configura um gerador simples baseado em templates."""
        self.use_simple = True
        self.generator = None
        self.tokenizer = None
        self.model = None

        # Templates para geração simples
        self.templates = {
            'historia': [
                "Era uma vez, em uma pequena cidade do interior do Brasil,",
                "Há muito tempo atrás, quando as tecnologias ainda não dominavam o mundo,",
                "Em um dia ensolarado de verão, algo extraordinário aconteceu:"
            ],
            'tecnologia': [
                "A tecnologia moderna tem revolucionado a forma como",
                "Com o avanço da inteligência artificial, agora é possível",
                "Os dispositivos móveis mudaram completamente nossa maneira de"
            ],
            'educacao': [
                "A educação no Brasil enfrenta diversos desafios, incluindo",
                "O ensino à distância se tornou uma realidade para milhões de estudantes que",
                "As metodologias ativas de aprendizagem permitem que os alunos"
            ]
        }
        print("✅ Gerador simples configurado!")

    def _simple_generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Geração simples baseada em templates e continuações lógicas.

        Args:
            prompt: Texto inicial
            max_length: Comprimento máximo aproximado

        Returns:
            Texto gerado
        """
        # Detectar categoria do prompt
        prompt_lower = prompt.lower()
        category = 'historia'  # padrão

        if any(word in prompt_lower for word in ['tecnologia', 'computador', 'internet', 'digital']):
            category = 'tecnologia'
        elif any(word in prompt_lower for word in ['educação', 'escola', 'ensino', 'estudante']):
            category = 'educacao'

        # Escolher template base
        if not prompt.strip():
            prompt = random.choice(self.templates[category])

        # Continuações genéricas
        continuations = [
            " que transformou a vida de muitas pessoas.",
            " e isso gerou consequências importantes para a sociedade.",
            " criando novas oportunidades e desafios únicos.",
            " permitindo avanços significativos na área.",
            " com resultados surpreendentes e promissores.",
            " estabelecendo novos padrões para o futuro."
        ]

        # Gerar texto simples
        base_text = prompt + random.choice(continuations)

        # Adicionar mais conteúdo se necessário
        if len(base_text.split()) < max_length // 3:
            base_text += " Este desenvolvimento representa um marco importante na história recente."

        return base_text

    def generate_text(self,
                     prompt: str,
                     max_length: int = MAX_LENGTH,
                     min_length: int = MIN_LENGTH,
                     temperature: float = TEMPERATURE,
                     top_k: int = TOP_K,
                     top_p: float = TOP_P,
                     num_return_sequences: int = 1,
                     do_sample: bool = True) -> Union[str, List[str]]:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: Texto inicial para geração
            max_length: Comprimento máximo do texto
            min_length: Comprimento mínimo do texto
            temperature: Controla aleatoriedade (0.1-2.0)
            top_k: Número de tokens mais prováveis a considerar
            top_p: Probabilidade cumulativa para nucleus sampling
            num_return_sequences: Número de sequências a gerar
            do_sample: Se deve usar amostragem

        Returns:
            Texto(s) gerado(s)
        """
        if not prompt:
            prompt = "Era uma vez"

        # Limpar prompt
        prompt = self._clean_prompt(prompt)

        try:
            # Verificar se deve usar gerador simples
            if hasattr(self, 'use_simple') and self.use_simple:
                results = []
                for _ in range(num_return_sequences):
                    text = self._simple_generate(prompt, max_length)
                    results.append(text)
                return results[0] if num_return_sequences == 1 else results

            if self.use_pipeline and self.generator:
                # Usar pipeline
                results = self.generator(
                    prompt,
                    max_length=max_length,
                    min_length=min_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=50256,  # GPT-2 padrão
                    truncation=True
                )

                # Extrair textos gerados
                generated_texts = [result['generated_text'] for result in results]
                return generated_texts[0] if num_return_sequences == 1 else generated_texts

            elif self.model and self.tokenizer:
                # Método manual
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Configurar seed para reprodutibilidade
                set_seed(42)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        min_length=min_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=do_sample,
                        repetition_penalty=REPETITION_PENALTY,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True
                    )

                # Decodificar textos
                generated_texts = []
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    generated_texts.append(text)

                return generated_texts[0] if num_return_sequences == 1 else generated_texts

            else:
                # Fallback para geração simples
                results = []
                for _ in range(num_return_sequences):
                    text = self._simple_generate(prompt, max_length)
                    results.append(text)
                return results[0] if num_return_sequences == 1 else results

        except Exception as e:
            print(f"⚠️ Erro na geração: {e}")
            print("🔄 Usando geração simples como fallback...")
            results = []
            for _ in range(num_return_sequences):
                text = self._simple_generate(prompt, max_length)
                results.append(text)
            return results[0] if num_return_sequences == 1 else results

    def _clean_prompt(self, prompt: str) -> str:
        """Limpa e prepara o prompt para geração."""
        # Remover quebras de linha excessivas
        prompt = re.sub(r'\n+', ' ', prompt)

        # Remover espaços excessivos
        prompt = re.sub(r'\s+', ' ', prompt)

        # Garantir que termine com espaço ou pontuação
        prompt = prompt.strip()
        if prompt and not prompt[-1] in '.!?:,':
            prompt += ' '

        return prompt

    def generate_creative_text(self,
                             theme: str,
                             style: str = "narrativo",
                             length: str = "medio") -> str:
        """
        Gera texto criativo baseado em tema e estilo.

        Args:
            theme: Tema do texto (ex: "aventura", "tecnologia", "amor")
            style: Estilo de escrita ("narrativo", "poetico", "jornalistico")
            length: Tamanho ("curto", "medio", "longo")

        Returns:
            Texto criativo gerado
        """
        # Definir prompts base por estilo
        style_prompts = {
            "narrativo": f"Era uma vez uma história sobre {theme}.",
            "poetico": f"Como pétalas ao vento, {theme} desperta em mim",
            "jornalistico": f"Em reportagem especial, investigamos {theme} e suas",
            "academico": f"Este estudo analisa {theme} considerando os aspectos",
            "conversacional": f"Você sabia que {theme} é muito interessante? Deixe-me contar"
        }

        # Definir comprimentos
        length_configs = {
            "curto": {"max_length": 80, "min_length": 30},
            "medio": {"max_length": 150, "min_length": 60},
            "longo": {"max_length": 300, "min_length": 120}
        }

        # Obter prompt e configurações
        prompt = style_prompts.get(style, style_prompts["narrativo"])
        config = length_configs.get(length, length_configs["medio"])

        # Ajustar parâmetros por estilo
        if style == "poetico":
            temperature = 1.1  # Mais criativo
            top_p = 0.95
        elif style == "jornalistico":
            temperature = 0.7  # Mais factual
            top_p = 0.85
        else:
            temperature = TEMPERATURE
            top_p = TOP_P

        print(f"✍️ Gerando texto {style} sobre '{theme}' ({length})...")

        return self.generate_text(
            prompt=prompt,
            max_length=config["max_length"],
            min_length=config["min_length"],
            temperature=temperature,
            top_p=top_p
        )

    def generate_multiple_variations(self,
                                   prompt: str,
                                   num_variations: int = 3,
                                   temperature_range: tuple = (0.6, 1.2)) -> List[Dict]:
        """
        Gera múltiplas variações de um mesmo prompt.

        Args:
            prompt: Prompt base
            num_variations: Número de variações
            temperature_range: Faixa de temperatura (min, max)

        Returns:
            Lista de variações com metadados
        """
        print(f"🎭 Gerando {num_variations} variações do prompt...")

        variations = []
        temp_min, temp_max = temperature_range

        for i in range(num_variations):
            # Variar parâmetros
            temp = temp_min + (temp_max - temp_min) * i / (num_variations - 1)
            top_k = random.randint(30, 80)
            top_p = random.uniform(0.8, 0.95)

            print(f"   Variação {i+1}/{num_variations} (temp={temp:.2f})...")

            try:
                text = self.generate_text(
                    prompt=prompt,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    max_length=180
                )

                variation = {
                    'variation_id': i + 1,
                    'text': text,
                    'temperature': temp,
                    'top_k': top_k,
                    'top_p': top_p,
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }

                variations.append(variation)

            except Exception as e:
                print(f"⚠️ Erro na variação {i+1}: {e}")
                continue

        return variations

    def analyze_generated_texts(self, texts: List[str]) -> Dict:
        """
        Analisa características dos textos gerados.

        Args:
            texts: Lista de textos para analisar

        Returns:
            Dicionário com estatísticas
        """
        if not texts:
            return {}

        stats = {
            'num_texts': len(texts),
            'word_counts': [len(text.split()) for text in texts],
            'char_counts': [len(text) for text in texts],
            'sentence_counts': [len(re.split(r'[.!?]+', text)) for text in texts],
            'avg_word_length': []
        }

        # Calcular comprimento médio das palavras
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            if words:
                avg_len = sum(len(word) for word in words) / len(words)
                stats['avg_word_length'].append(avg_len)
            else:
                stats['avg_word_length'].append(0)

        # Estatísticas agregadas
        stats['total_words'] = sum(stats['word_counts'])
        stats['avg_words_per_text'] = np.mean(stats['word_counts'])
        stats['avg_chars_per_text'] = np.mean(stats['char_counts'])
        stats['avg_sentences_per_text'] = np.mean(stats['sentence_counts'])
        stats['avg_word_length_overall'] = np.mean(stats['avg_word_length'])

        return stats
# %%
generator = TextGenerator(
    model_name=MODELO_PRINCIPAL,
    use_pipeline=True
)
summarizer = PortugueseSummarizer()

# %%
prompt_teste = "A inteligência artificial no Brasil"
texto_gerado = generator.generate_text(prompt_teste, max_length=120)
texto_gerado
# %%

memoria = ""
while True:
  msg = input("Digite sua msg:")
  memoria += "\nHuman: "+ msg
  texto_gerado = generator.generate_text(memoria, max_length=120)
  print(texto_gerado)
  memoria += "\nAI: " + texto_gerado
  memoria += "\n"
  memoria = summarizer.summarize_text(
      memoria,
      max_length=300
  )
  if msg == "sair":
    break
# %%
