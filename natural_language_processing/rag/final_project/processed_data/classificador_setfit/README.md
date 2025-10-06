---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: O que é publicidade enganosa segundo CDC?
- text: Quais direitos do consumidor?
- text: Quanto é 30 dias de aviso prévio?
- text: Bom dia
- text: Quem criou você?
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: rufimelo/Legal-BERTimbau-base
---

# SetFit with rufimelo/Legal-BERTimbau-base

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [rufimelo/Legal-BERTimbau-base](https://huggingface.co/rufimelo/Legal-BERTimbau-base) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [rufimelo/Legal-BERTimbau-base](https://huggingface.co/rufimelo/Legal-BERTimbau-base)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                | Examples                                                                                                                               |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| rag_required         | <ul><li>'O que diz o artigo 18 do CDC?'</li><li>'Quais direitos do consumidor?'</li><li>'Como funciona recall?'</li></ul>              |
| general_conversation | <ul><li>'Olá, tudo bem?'</li><li>'Oi, como você está?'</li><li>'O que você pode fazer?'</li></ul>                                      |
| calculation          | <ul><li>'Calcule minhas férias de 8 meses'</li><li>'Quanto é 30 dias de aviso prévio?'</li><li>'Qual o prazo para reclamar?'</li></ul> |
| out_of_scope         | <ul><li>'Receita de bolo de chocolate'</li><li>'Como tocar violão?'</li><li>'Previsão do tempo'</li></ul>                              |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Bom dia")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 2   | 4.375  | 8   |

| Label                | Training Sample Count |
|:---------------------|:----------------------|
| rag_required         | 8                     |
| general_conversation | 8                     |
| calculation          | 8                     |
| out_of_scope         | 8                     |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: 50
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.02  | 1    | 0.1723        | -               |
| 1.0   | 50   | 0.1657        | -               |

### Framework Versions
- Python: 3.12.11
- SetFit: 1.1.3
- Sentence Transformers: 5.1.1
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu126
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->