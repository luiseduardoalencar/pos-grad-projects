#%%
#TEXT_CLASSIFICATION.PY
#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")
#%%
# Configurações globais
MODELO_PRETRAINED = "neuralmind/bert-base-portuguese-cased"  # Modelo PT-BR
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42
#%%
class TextClassificationDataset(Dataset):
    """
    Dataset personalizado para classificação de texto.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Inicializa o dataset.

        Args:
            texts: Lista de textos
            labels: Lista de labels
            tokenizer: Tokenizer do transformers
            max_length: Comprimento máximo da sequência
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenizar o texto
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
#%%
class TextClassifier:
    """
    Classificador de texto usando BERT português.
    """

    def __init__(self, model_name=MODELO_PRETRAINED, num_classes=8):
        """
        Inicializa o classificador.

        Args:
            model_name: Nome do modelo pré-treinado
            num_classes: Número de classes para classificação
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 Inicializando classificador...")
        print(f"   • Modelo: {model_name}")
        print(f"   • Classes: {num_classes}")
        print(f"   • Dispositivo: {self.device}")

        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Carregar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        self.model.to(self.device)

        # Label encoder
        self.label_encoder = LabelEncoder()

        print("✅ Classificador inicializado com sucesso!")

    def prepare_data(self, df, test_size=0.2):
        """
        Prepara os dados para treinamento.

        Args:
            df: DataFrame com colunas 'text' e 'label'
            test_size: Proporção dos dados para teste

        Returns:
            DataLoaders para treino e teste
        """
        print(f"📊 Preparando dados...")
        print(f"   • Total de amostras: {len(df)}")

        # Remover duplicatas
        df_clean = df.drop_duplicates().reset_index(drop=True)
        print(f"   • Após remoção de duplicatas: {len(df_clean)}")

        # Separar textos e labels
        texts = df_clean['text'].tolist()
        labels = df_clean['label'].tolist()

        # Codificar labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Mostrar mapeamento de classes
        print(f"\n🏷️ Mapeamento de classes:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = sum(1 for label in labels if label == class_name)
            print(f"   • {i}: {class_name} ({count} amostras)")

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=encoded_labels
        )

        print(f"\n📚 Divisão dos dados:")
        print(f"   • Treino: {len(X_train)} amostras")
        print(f"   • Teste: {len(X_test)} amostras")

        # Criar datasets
        train_dataset = TextClassificationDataset(
            X_train, y_train, self.tokenizer, MAX_LENGTH
        )
        test_dataset = TextClassificationDataset(
            X_test, y_test, self.tokenizer, MAX_LENGTH
        )

        # Criar DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # Salvar dados de teste para avaliação posterior
        self.X_test = X_test
        self.y_test = y_test

        return train_loader, test_loader

    def train(self, train_loader, test_loader, epochs=EPOCHS):
        """
        Treina o modelo.

        Args:
            train_loader: DataLoader de treino
            test_loader: DataLoader de teste
            epochs: Número de épocas
        """
        print(f"\n🚀 Iniciando treinamento...")
        print(f"   • Épocas: {epochs}")
        print(f"   • Learning rate: {LEARNING_RATE}")
        print(f"   • Batch size: {BATCH_SIZE}")

        # Configurar otimizador
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # Scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Listas para armazenar métricas
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            print(f"\n📚 Época {epoch + 1}/{epochs}")

            # Treinamento
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(train_loader, desc=f"Treinando")

            for batch in progress_bar:
                # Mover dados para dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Zerar gradientes
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Calcular acurácia
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                total_loss += loss.item()

                # Atualizar barra de progresso
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}'
                })

            # Métricas da época
            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            print(f"   📊 Loss médio: {avg_loss:.4f}")
            print(f"   🎯 Acurácia: {accuracy:.4f}")

            # Avaliação no conjunto de teste
            test_accuracy = self.evaluate(test_loader, verbose=False)
            print(f"   🧪 Acurácia no teste: {test_accuracy:.4f}")

        print("\n✅ Treinamento concluído!")

        # Plotar curvas de treinamento
        self._plot_training_curves(train_losses, train_accuracies)

        return train_losses, train_accuracies

    def evaluate(self, test_loader, verbose=True):
        """
        Avalia o modelo no conjunto de teste.

        Args:
            test_loader: DataLoader de teste
            verbose: Se deve imprimir detalhes

        Returns:
            Acurácia do modelo
        """
        if verbose:
            print(f"\n🧪 Avaliando modelo...")

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct_predictions / total_predictions

        if verbose:
            print(f"   🎯 Acurácia final: {accuracy:.4f}")

            # Relatório de classificação
            class_names = self.label_encoder.classes_
            report = classification_report(
                all_labels, all_predictions,
                target_names=class_names,
                output_dict=True
            )

            print(f"\n📊 Relatório de Classificação:")
            print(classification_report(all_labels, all_predictions, target_names=class_names))

            # Matriz de confusão
            self._plot_confusion_matrix(all_labels, all_predictions, class_names)

        return accuracy

    def predict(self, texts):
        """
        Faz predições para novos textos.

        Args:
            texts: Lista de textos ou texto único

        Returns:
            Predições e probabilidades
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=MAX_LENGTH,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)

                # Converter para classe original
                pred_class = self.label_encoder.inverse_transform([prediction.cpu().item()])[0]
                pred_prob = probs.cpu().numpy()[0]

                predictions.append(pred_class)
                probabilities.append(pred_prob)

        return predictions, probabilities

    def _plot_training_curves(self, losses, accuracies):
        """Plota curvas de treinamento."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(losses, 'b-', label='Loss de Treino')
        ax1.set_title('📉 Loss durante Treinamento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Acurácia
        ax2.plot(accuracies, 'g-', label='Acurácia de Treino')
        ax2.set_title('📈 Acurácia durante Treinamento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("💾 Curvas de treinamento salvas como 'training_curves.png'")
        plt.show()

    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plota matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('🎯 Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.xlabel('Predição', fontsize=12)
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("💾 Matriz de confusão salva como 'confusion_matrix.png'")
        plt.show()

    def save_model(self, path='./modelo_classificador'):
        """
        Salva o modelo treinado.

        Args:
            path: Caminho para salvar o modelo
        """
        print(f"💾 Salvando modelo em {path}...")

        # Criar diretório se não existir
        os.makedirs(path, exist_ok=True)

        # Salvar modelo e tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Salvar label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.pkl'))

        print("✅ Modelo salvo com sucesso!")

    def load_model(self, path='./modelo_classificador'):
        """
        Carrega um modelo salvo.

        Args:
            path: Caminho do modelo salvo
        """
        print(f"📂 Carregando modelo de {path}...")

        # Carregar modelo e tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

        # Carregar label encoder
        import joblib
        self.label_encoder = joblib.load(os.path.join(path, 'label_encoder.pkl'))

        print("✅ Modelo carregado com sucesso!")

# %%
# Se você está rodando de dentro de classification/
df = pd.read_csv('../dataset_generator/meu_dataset.csv')
num_classes = df['label'].nunique()
classifier = TextClassifier(num_classes=num_classes)
# %%
train_loader, test_loader = classifier.prepare_data(df)
# %%
train_losses, train_accuracies = classifier.train(train_loader, test_loader)
# %%
final_accuracy = classifier.evaluate(test_loader)
# %%
# Teste com exemplos novos
exemplos_teste = [
    "Governo anuncia nova política de saúde pública",
    "Flamengo vence o clássico por 2 a 1",
    "Dólar sobe e bolsa cai nesta segunda-feira",
    "Nova tecnologia de inteligência artificial é desenvolvida"
]
predictions, probabilities = classifier.predict(exemplos_teste)
for i, (texto, pred, probs) in enumerate(zip(exemplos_teste, predictions, probabilities)):
  print(f"\n{i+1}. Texto: {texto}")
  print(f"   Predição: {pred}")
  print(f"   Confiança: {max(probs):.3f}")

  # Mostrar top 3 classes mais prováveis
  top_3_idx = np.argsort(probs)[-3:][::-1]
  print(f"   Top 3 classes:")
  for idx in top_3_idx:
      class_name = classifier.label_encoder.classes_[idx]
      prob = probs[idx]
      print(f"     • {class_name}: {prob:.3f}")
# %%
classifier.save_model()
# %%
