# 🚀 Guia de Configuração - Ambiente GPU para NLP

Este guia contém todos os comandos necessários para configurar um ambiente Python com suporte a GPU NVIDIA para projetos de Processamento de Linguagem Natural (NLP).

## 📋 Pré-requisitos

- Anaconda ou Miniconda instalado
- GPU NVIDIA (testado com RTX 4050)
- Driver NVIDIA atualizado (versão 581.42 ou superior)
- Windows 10/11

## 🎯 Informações do Ambiente

- **Nome do ambiente**: `nlp_gpu`
- **Versão Python**: 3.12.11
- **PyTorch**: versão mais recente com CUDA 12.1
- **GPU suportada**: NVIDIA GeForce RTX 4050 (6GB VRAM)

---

## 🔧 Instalação Passo a Passo

### 1️⃣ Criar o Ambiente Conda

```bash
conda create -n nlp_gpu python=3.12 -y
```

### 2️⃣ Ativar o Ambiente

```bash
conda activate nlp_gpu
```

### 3️⃣ Verificar o Caminho do Python

```bash
where python
```

**Resultado esperado:**
```
C:\user\user_name\anaconda3\envs\nlp_gpu\python.exe
```

⚠️ **IMPORTANTE**: Copie este caminho completo para usar nos próximos comandos!

### 4️⃣ Instalar PyTorch com Suporte GPU

```bash
C:\user\user_name\anaconda3\envs\nlp_gpu\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🖥️ Configurar no VSCode

1. Abrir arquivo `.py` ou notebook
2. Clicar no **seletor de kernel** (canto superior direito)
3. Selecionar **"Python 3.12 (NLP GPU)"**
4. Pronto! 🎉

### Usar dentro do código Python

Depois de selecionar o kernel correto, você pode instalar pacotes diretamente no código:

```python
#%%
%pip install nome-do-pacote
```

---

## ✅ Testar a Configuração

### Teste Básico

```python
import torch

print("=" * 50)
print(f"🎮 CUDA disponível: {torch.cuda.is_available()}")
print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
print(f"⚡ Versão CUDA: {torch.version.cuda}")
print(f"💾 Memória total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 50)
```

### Teste com Transformers

```python
from transformers import pipeline

# Criar pipeline de NER com GPU
ner = pipeline("ner", model="pierreguillou/ner-bert-base-cased-pt-lenerbr", device=0)

# Testar
texto = "João mora em São Paulo e trabalha na Microsoft."
resultado = ner(texto)

print(resultado)
print(f"\n📊 Memória GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

### Monitorar GPU em Tempo Real

Abra um terminal separado e execute:

```bash
nvidia-smi -l 1
```

Isso atualiza a cada 1 segundo mostrando uso de GPU e memória.

---

## 🔄 Comandos Úteis

### Ativar ambiente

```bash
conda activate nlp_gpu
```

### Desativar ambiente

```bash
conda deactivate
```



### Listar pacotes instalados

```bash
C:\Users\luis\anaconda3\envs\nlp_gpu\python.exe -m pip list
```

### Remover ambiente (se necessário)

```bash
conda deactivate
conda env remove -n nlp_gpu
```

---

## 🐛 Troubleshooting

### Problema: `CUDA disponível: False`

**Solução:**
1. Verifique se instalou usando o caminho completo do python.exe
2. Reinstale o PyTorch:
```bash
C:\Users\luis\anaconda3\envs\nlp_gpu\python.exe -m pip uninstall torch torchvision torchaudio -y
C:\Users\luis\anaconda3\envs\nlp_gpu\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problema: Pacotes instalando fora do ambiente

**Solução:**
- Sempre use o caminho completo: `C:\Users\luis\anaconda3\envs\nlp_gpu\python.exe -m pip install`
- OU ative o ambiente e use: `python -m pip install` (não apenas `pip install`)

### Problema: Kernel não aparece no VSCode

**Solução:**
1. Reinstale o kernel:
```bash
C:\Users\luis\anaconda3\envs\nlp_gpu\python.exe -m ipykernel install --user --name=nlp_gpu --display-name "Python 3.12 (NLP GPU)"
```
2. Reinicie o VSCode
3. Clique em "Select Kernel" e escolha "Python 3.12 (NLP GPU)"

---

## 📊 Especificações Testadas

- **Sistema Operacional**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Memória GPU**: 6141 MB
- **Driver NVIDIA**: 581.42
- **CUDA Version**: 13.0 (driver) / 12.1 (PyTorch)
- **Python**: 3.12.11
- **PyTorch**: 2.8.0+cu121

---

## 📝 Notas Importantes

1. **Python 3.13 não é suportado**: PyTorch ainda não tem suporte oficial para Python 3.13. Use Python 3.12!

2. **Sempre use o caminho completo**: Para garantir que os pacotes sejam instalados no ambiente correto, use sempre o caminho completo do python.exe do ambiente.

3. **Compatibilidade CUDA**: Seu driver suporta CUDA 13.0, mas o PyTorch usa CUDA 12.1 (compatível).

4. **Memória GPU**: Com 6GB de VRAM, você pode processar modelos BERT base. Para modelos maiores (large/XL), pode precisar reduzir batch size.

---

## 📚 Recursos Adicionais

- [Documentação PyTorch](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)

---

## ✨ Pronto para Usar!

Agora você tem um ambiente completo com suporte GPU para seus projetos de NLP! 🎉

**Happy coding!** 🚀
