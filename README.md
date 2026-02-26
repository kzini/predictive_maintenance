# Manutenção preditiva com machine learning  
## Prognóstico de falhas em motores utilizando o dataset CMAPSS (NASA)

Este projeto implementa uma pipeline completa de predição de **RUL (Remaining Useful Life)** utilizando o dataset CMAPSS (NASA), com foco em generalização real, validação por grupos e modelagem da degradação temporal.

---

## Objetivo

Estimar a vida útil remanescente de motores a partir de dados multissensoriais, respeitando:

- dependência temporal  
- estrutura por unidade (motor)  
- degradação progressiva  
- generalização entre motores distintos

---

## Pipeline

- Análise exploratória orientada à vida útil  
- Limpeza e seleção de sensores  
- Feature engineering temporal 
- Construção do target RUL  
- Validação por grupos (`GroupKFold`)  
- Modelagem preditiva  
- Otimização de hiperparâmetros  

---

## Feature engineering temporal

As seguintes técnicas foram utilizadas:

- **Médias móveis** - Com o objetivo de suavizar ruídos dos sensores, capturar tendências de longo prazo e reduzir impacto de picos momentâneos;

- **Variação absoluta em múltiplas escalas** - Para capturar taxa de degradação, detectar mudanças abruptas e padrões de curto, médio e longo prazo;

- **Variação percentual** - Para normalizar variações,  comparar taxa de degradação e detectar aceleração no processo de falha;

- **Health Index** - Consolida múltiplos sensores em uma única feature, simplificando o monitoramento e a interpretação da degradação.

---

## Modelos avaliados

- Random Forest  
- XGBoost  
- LightGBM  

---

## Metodologia de avaliação

- Split por motor (evitando vazamento temporal e estrutural entre ciclos do mesmo equipamento) 
- Validação cruzada com `GroupKFold`  
- Comparação entre:
  - validação simples por grupo (holdout estruturado)
  - validação cruzada por grupo (robustez de generalização)
- Métricas: RMSE e R²  
- Análise de gap entre validação simples e CV como critério de estabilidade

---

## Modelo final

**XGBoost com Randomized Search**  

Selecionado não pelo menor RMSE absoluto, mas pelo menor gap entre validação direta e cross-validation, indicando melhor capacidade de generalização estrutural.

---

## Interpretabilidade

A análise de importância de features (SHAP) indica o `health_index_ma10` disparadamente como a variável mais relevante, mostrando que o modelo aprende padrões latentes de degradação, e não apenas sinais brutos de sensores isolados. As demais features relevantes são majoritariamente transformações temporais (ma, diff), evidenciando aprendizado multiescala da degradação ao longo do tempo.

---

## Resultados (ordem de grandeza)

- RMSE: 11.24
- R² (porcentagem): 92.74%  

---

## Nota metodológica

- A remoção de sensores não foi feita por critério puramente automático. Primeiro, foi realizada análise visual dos sinais ao longo do tempo. Em seguida, foi aplicado um método de detecção de features constantes. Salvo os sensores 9 e 14, ambos os métodos convergiram para o mesmo conjunto de sensores, reforçando a validade da decisão de exclusão.

- Não foi definido um conjunto de teste final cego separado, pois o foco do projeto é a avaliação da generalização estrutural entre motores, por meio de validação por grupos e validação cruzada por entidade (GroupKFold), e não benchmarking competitivo nem estimativa de performance absoluta final. A combinação de holdout estruturado por motor e cross-validation por grupo permitiu analisar robustez, estabilidade e capacidade de generalização entre unidades.

---

## Estrutura do projeto

```
predictive_maintenance
├── src/
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── utils.py
│   └── visualization.py
├── notebook/
│   └── predictive_maintenance.ipynb
└── README.md

```
---

## Como Reproduzir

### 1. Clone o repositório:
```bash
git clone https://github.com/kzini/predictive_maintenance
cd predictive_maintenance
```

### 2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 3. Execute o notebook:
notebook/predictive_maintenance.ipynb  

---

## Autor

**Bruno Casini**  
LinkedIn: https://www.linkedin.com/in/kzini
