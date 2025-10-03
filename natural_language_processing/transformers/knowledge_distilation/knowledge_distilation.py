#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    pipeline
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import warnings
import os
from dataclasses import dataclass

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
# %%
# Configurações
TEACHER_MODEL = "neuralmind/bert-base-portuguese-cased"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 4.0  # Temperatura para suavizar distribuições
ALPHA = 0.7        # Peso da loss de destilação vs loss hard
# %%
@dataclass
class DistillationConfig:
    """Configuração para destilação de conhecimento."""
    teacher_model_name: str = TEACHER_MODEL
    temperature: float = TEMPERATURE
    alpha: float = ALPHA
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
# %%
class KnowledgeDistillationTrainer(Trainer):
    """
    Trainer customizado para Knowledge Distillation.

    Implementa a loss de destilação que combina:
    1. Hard targets: Loss padrão com rótulos verdadeiros
    2. Soft targets: Loss de destilação com saídas do professor
    """

    def __init__(self, teacher_model, temperature=4.0, alpha=0.7, **kwargs):
        """
        Inicializa o trainer de destilação.

        Args:
            teacher_model: Modelo professor pré-treinado
            temperature: Temperatura para suavizar distribuições
            alpha: Peso da loss de destilação (1-alpha = peso da loss hard)
            **kwargs: Argumentos do Trainer base
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha

        # Mover teacher para o mesmo dispositivo
        if hasattr(self.model, 'device'):
            self.teacher_model.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computa a loss combinada de destilação.

        Args:
            model: Modelo estudante
            inputs: Batch de dados de entrada
            return_outputs: Se deve retornar as saídas

        Returns:
            Loss combinada ou (loss, outputs)
        """
        # Forward pass do estudante
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Forward pass do professor
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Loss hard (com rótulos verdadeiros)
        if "labels" in inputs:
            loss_ce = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                inputs["labels"].view(-1),
                ignore_index=-100
            )
        else:
            loss_ce = 0

        # Loss de destilação (soft targets)
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # Loss combinada
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce

        return (loss, student_outputs) if return_outputs else loss
# %%
class PortugueseDistillation:
    """
    Sistema completo de Knowledge Distillation para modelos portugueses.
    """

    def __init__(self, config: DistillationConfig):
        """
        Inicializa o sistema de destilação.

        Args:
            config: Configurações de destilação
        """
        self.config = config
        self.device = DEVICE

        print(f"🧠 KNOWLEDGE DISTILLATION PARA PORTUGUÊS")
        print(f"   • Modelo professor: {config.teacher_model_name}")
        print(f"   • Dispositivo: {self.device}")
        print(f"   • Temperatura: {config.temperature}")
        print(f"   • Alpha: {config.alpha}")
        print("=" * 60)

        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None

    def load_teacher_model(self):
        """Carrega o modelo professor."""
        print("👨‍🏫 Carregando modelo professor...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.teacher_model_name)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model_name,
                torch_dtype=torch.float32
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Análise do modelo professor
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            print(f"✅ Modelo professor carregado!")
            print(f"   • Parâmetros: {teacher_params:,}")
            print(f"   • Tamanho: ~{teacher_params / 1e6:.1f}M parâmetros")

            return True

        except Exception as e:
            print(f"❌ Erro ao carregar professor: {e}")
            return False

    def create_student_model(self, reduction_factor: float = 0.5):
        """
        Cria modelo estudante menor baseado no professor.

        Args:
            reduction_factor: Fator de redução (0.5 = 50% menor)
        """
        print(f"👨‍🎓 Criando modelo estudante ({reduction_factor*100}% do tamanho)...")

        if self.teacher_model is None:
            print("❌ Carregue o modelo professor primeiro!")
            return False

        try:
            # Obter configuração do professor
            teacher_config = self.teacher_model.config

            # Calcular arquitetura reduzida
            student_config = AutoConfig.from_pretrained(
                self.config.teacher_model_name,
                vocab_size=teacher_config.vocab_size,
                n_positions=getattr(teacher_config, 'n_positions', 1024),
                n_embd=max(64, int(teacher_config.n_embd * reduction_factor)),
                n_layer=max(1, int(teacher_config.n_layer * reduction_factor)),
                n_head=max(1, int(teacher_config.n_head * reduction_factor)),
                resid_pdrop=teacher_config.resid_pdrop,
                embd_pdrop=teacher_config.embd_pdrop,
                attn_pdrop=teacher_config.attn_pdrop,
                pad_token_id=teacher_config.pad_token_id,
                eos_token_id=teacher_config.eos_token_id
            )

            # Criar modelo estudante
            self.student_model = AutoModelForCausalLM.from_config(
                student_config
            ).to(self.device)

            # Análise comparativa
            student_params = sum(p.numel() for p in self.student_model.parameters())
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            reduction = (1 - student_params / teacher_params) * 100

            print(f"✅ Modelo estudante criado!")
            print(f"   • Camadas: {teacher_config.n_layer} → {student_config.n_layer}")
            print(f"   • Dimensão: {teacher_config.n_embd} → {student_config.n_embd}")
            print(f"   • Cabeças de atenção: {teacher_config.n_head} → {student_config.n_head}")
            print(f"   • Parâmetros: {teacher_params:,} → {student_params:,}")
            print(f"   • Redução: {reduction:.1f}%")

            return True

        except Exception as e:
            print(f"❌ Erro ao criar estudante: {e}")
            return False

    def prepare_training_data(self, texts: List[str], max_length: int = 512):
        """
        Prepara dados para treinamento de destilação.

        Args:
            texts: Lista de textos para treinamento
            max_length: Comprimento máximo das sequências

        Returns:
            Dataset tokenizado
        """
        print(f"📚 Preparando dados de treinamento...")
        print(f"   • Textos: {len(texts)}")
        print(f"   • Max length: {max_length}")

        # Tokenizar textos
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Criar dataset simples
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': self.encodings['input_ids'][idx].clone()
                }

            def __len__(self):
                return len(self.encodings['input_ids'])

        dataset = SimpleDataset(encodings)
        print(f"✅ Dataset preparado: {len(dataset)} exemplos")

        return dataset

    def train_student(self, train_dataset, eval_dataset=None, output_dir="./distilled_model"):
        """
        Treina o modelo estudante usando destilação.

        Args:
            train_dataset: Dataset de treinamento
            eval_dataset: Dataset de validação (opcional)
            output_dir: Diretório para salvar o modelo
        """
        print(f"🎓 Iniciando treinamento com destilação de conhecimento...")

        if self.teacher_model is None or self.student_model is None:
            print("❌ Carregue os modelos professor e estudante primeiro!")
            return False

        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=[]
        )

        # Criar trainer de destilação
        trainer = KnowledgeDistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            model=self.student_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        try:
            # Treinar modelo
            print("🚀 Iniciando treinamento...")
            trainer.train()

            # Salvar modelo final
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            print(f"✅ Treinamento concluído!")
            print(f"   • Modelo salvo em: {output_dir}")

            return True

        except Exception as e:
            print(f"❌ Erro durante treinamento: {e}")
            return False

    def evaluate_models(self, test_texts: List[str], max_length: int = 100):
        """
        Compara performance dos modelos professor e estudante.

        Args:
            test_texts: Textos para teste
            max_length: Comprimento máximo para geração

        Returns:
            Dicionário com métricas de comparação
        """
        print(f"📊 Avaliando modelos professor vs estudante...")

        if self.teacher_model is None or self.student_model is None:
            print("❌ Modelos não disponíveis para avaliação!")
            return {}

        results = {
            'teacher': {'times': [], 'outputs': []},
            'student': {'times': [], 'outputs': []},
            'prompts': test_texts
        }

        # Avaliar modelo professor
        print("   Avaliando professor...")
        self.teacher_model.eval()
        for text in tqdm(test_texts):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            end_time = time.time()

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results['teacher']['times'].append(end_time - start_time)
            results['teacher']['outputs'].append(generated_text)

        # Avaliar modelo estudante
        print("   Avaliando estudante...")
        self.student_model.eval()
        for text in tqdm(test_texts):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = self.student_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            end_time = time.time()

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results['student']['times'].append(end_time - start_time)
            results['student']['outputs'].append(generated_text)

        # Calcular métricas
        teacher_avg_time = np.mean(results['teacher']['times'])
        student_avg_time = np.mean(results['student']['times'])
        speedup = teacher_avg_time / student_avg_time

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        size_reduction = (1 - student_params / teacher_params) * 100

        summary = {
            'teacher_avg_time': teacher_avg_time,
            'student_avg_time': student_avg_time,
            'speedup': speedup,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'size_reduction': size_reduction
        }

        print(f"📈 Resultados da Avaliação:")
        print(f"   • Tempo médio professor: {teacher_avg_time:.3f}s")
        print(f"   • Tempo médio estudante: {student_avg_time:.3f}s")
        print(f"   • Aceleração: {speedup:.2f}x")
        print(f"   • Redução de tamanho: {size_reduction:.1f}%")

        results.update(summary)
        return results

    def create_comparison_dashboard(self, evaluation_results: Dict):
        """
        Cria dashboard visual comparando os modelos.

        Args:
            evaluation_results: Resultados da avaliação
        """
        print(f"📊 Criando dashboard de comparação...")

        if not evaluation_results:
            print("❌ Nenhum resultado disponível!")
            return

        # Preparar dados
        models = ['Professor', 'Estudante']
        times = [evaluation_results['teacher_avg_time'], evaluation_results['student_avg_time']]
        params = [evaluation_results['teacher_params']/1e6, evaluation_results['student_params']/1e6]

        # Criar dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Tempo de inferência
        bars1 = ax1.bar(models, times, color=['red', 'blue'], alpha=0.7)
        ax1.set_title('⏱️ Tempo de Inferência', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Tempo (segundos)')
        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')

        # 2. Número de parâmetros
        bars2 = ax2.bar(models, params, color=['red', 'blue'], alpha=0.7)
        ax2.set_title('📊 Tamanho do Modelo', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Parâmetros (Milhões)')
        for bar, param_val in zip(bars2, params):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{param_val:.1f}M', ha='center', va='bottom', fontweight='bold')

        # 3. Métricas de eficiência
        metrics = ['Aceleração', 'Redução Tamanho (%)']
        values = [evaluation_results['speedup'], evaluation_results['size_reduction']]
        colors = ['green', 'orange']

        bars3 = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_title('🚀 Métricas de Eficiência', fontweight='bold', fontsize=14)
        for bar, val in zip(bars3, values):
            height = bar.get_height()
            if 'Aceleração' in metrics[bars3.index(bar)]:
                label = f'{val:.2f}x'
            else:
                label = f'{val:.1f}%'
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontweight='bold')

        # 4. Comparação de textos gerados (primeiros 3 exemplos)
        ax4.axis('off')
        ax4.set_title('📝 Comparação de Saídas', fontweight='bold', fontsize=14)

        y_pos = 0.9
        for i in range(min(3, len(evaluation_results['prompts']))):
            prompt = evaluation_results['prompts'][i][:30] + "..."
            teacher_out = evaluation_results['teacher']['outputs'][i][:50] + "..."
            student_out = evaluation_results['student']['outputs'][i][:50] + "..."

            ax4.text(0.05, y_pos, f"Prompt {i+1}: {prompt}", fontsize=10, fontweight='bold')
            ax4.text(0.05, y_pos-0.05, f"Professor: {teacher_out}", fontsize=9, color='red')
            ax4.text(0.05, y_pos-0.1, f"Estudante: {student_out}", fontsize=9, color='blue')
            y_pos -= 0.25

        plt.tight_layout()
        plt.savefig('knowledge_distillation_comparison.png', dpi=300, bbox_inches='tight')
        print("💾 Dashboard salvo como 'knowledge_distillation_comparison.png'")
        plt.show()

# %%
config = DistillationConfig(
        teacher_model_name="pierreguillou/gpt2-small-portuguese",
        temperature=4.0,
        alpha=0.7,
        num_train_epochs=1,  # Reduzido para demonstração
        per_device_train_batch_size=4
    )
distiller = PortugueseDistillation(config)
# %%
distiller.load_teacher_model()
distiller.create_student_model(reduction_factor=0.5)
# %%
training_texts = [
        "A inteligência artificial está revolucionando o mundo",
        "O Brasil é um país com grande potencial tecnológico",
        "A educação é fundamental para o desenvolvimento da sociedade",
        "Machine learning pode ajudar na resolução de problemas complexos",
        "A tecnologia deve ser usada para beneficiar toda a humanidade",
        "Processamento de linguagem natural é uma área fascinante",
        "Modelos de linguagem podem gerar textos muito interessantes",
        "A pesquisa em IA avança rapidamente no século XXI"
    ]
train_dataset = distiller.prepare_training_data(training_texts, max_length=128)
# %%
distiller.train_student(train_dataset)
# %%
test_texts = [
        "O futuro da tecnologia será",
        "A IA pode ajudar",
        "Aprendizado de máquina está realizando"
    ]
results = distiller.evaluate_models(test_texts, max_length=50)
# %%
distiller.create_comparison_dashboard(results)
# %%
