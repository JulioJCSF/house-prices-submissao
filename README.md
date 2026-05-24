# 🏠 House Prices Prediction — Repositório de Submissão

Repositório de submissão do **Projeto 2 — Competição de Machine Learning: Previsão de Preços de Imóveis** da equipe abaixo. Contém apenas o código de execução automatizada que será chamado pelo corretor.

## 👥 Equipe

| Nome | Matrícula |
|---|---|
| Cauã Bilhar Dacca | 2315904 |
| Julio Cesar de Sousa Fernandes | 2316701 |
| Roberto Nascimento Xavier | 2316446 |
| Thiago Neves de Carvalho | 2316921 |

**Disciplina:** Machine Learning
**Professor:** Tulio Rodrigues Ribeiro

## 🔗 Repositórios

- **Submissão (este):** [`house-prices-submissao`](https://github.com/JulioJCSF/house-prices-submissao)
- **Desenvolvimento (EDA, scripts, notebooks):** [`house-prices-desenvolvimento`](https://github.com/JulioJCSF/house-prices-desenvolvimento)

> ⚠️ Atualizem os links acima depois de subir os repos no GitHub.

## 📦 Estrutura

```
.
├── pipeline.py          # Função prever_precos() chamada pelo corretor
├── modelo.joblib        # Pipeline scikit-learn treinado (XGBoost)
├── requirements.txt     # Dependências mínimas
└── README.md            # Este arquivo
```

## 🚀 Como executar

```bash
pip install -r requirements.txt
python pipeline.py
```

O script lê `teste_publico.csv` (caso esteja na mesma pasta), gera as predições e imprime as 5 primeiras. Para uso programático, basta importar:

```python
from pipeline import prever_precos
predicoes = prever_precos('caminho/para/teste.csv')   # retorna np.array em dólares
```

## 🧠 Modelo

- **Algoritmo final:** XGBoost Regressor (`n_estimators=1000, max_depth=4, learning_rate=0.05`)
- **Target transformada:** `np.log1p(SalePrice)` — convertida de volta com `np.expm1` antes do retorno
- **Pré-processamento (integrado no Pipeline scikit-learn):**
  - Imputação semântica de NaN ("None" para categóricas e 0 para numéricas onde NaN significa ausência da feature)
  - `LotFrontage` preenchido pela mediana do bairro
  - Encoding ordinal manual para variáveis de qualidade (`Ex=5, Gd=4, ...`)
  - One-Hot Encoding (`handle_unknown='ignore'`) para demais categóricas
  - StandardScaler nas numéricas
  - Feature engineering: `TotalSF`, `HouseAge`, `RemodAge`

## 📊 Desempenho

Validação cruzada 5-fold no `treino.csv` (random_state=42):

| Métrica | Valor |
|---|---|
| RMSLE | **0.11794** ± 0.00445 |
| MAE | **\$14.240** |
| R² | **0.910** |
| Baseline do professor (RMSLE) | 0.17543 |
| **Melhora sobre o baseline** | **+32,8%** |

Detalhes da comparação entre os 3 modelos testados (Regressão Linear, Random Forest, XGBoost) estão no **Repositório de Desenvolvimento**.

## ⏱️ Tempo de execução

Pipeline completo (carregar modelo + processar 1459 imóveis + retornar predições): **~7 segundos** em hardware comum. Limite estipulado pelo professor: 60s.

## 🐍 Compatibilidade

Modelo salvo em scikit-learn 1.8.0, joblib 1.5.3, Python 3.12. Testado e compatível com Python 3.10 – 3.13.
