"""
Pipeline de predição — House Prices
====================================
Função obrigatória prever_precos(caminho_arquivo_teste) que será chamada
pelo corretor automático. Recebe o caminho do CSV de teste e retorna
um np.array com as predições em dólares.
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_log_error

warnings.filterwarnings('ignore')


# =====================================================================
# CONSTANTES DO PRÉ-PROCESSAMENTO (estratégia definida no EDA)
# =====================================================================
COLS_NA_NONE = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType',
]

COLS_NA_ZERO = [
    'GarageYrBlt', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath',
    'GarageCars', 'GarageArea',
]

COLS_ORDINAL_QUAL = [
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond',
    'HeatingQC',
    'KitchenQual',
    'FireplaceQu',
    'GarageQual', 'GarageCond',
    'PoolQC',
]
MAP_QUAL = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}


# =====================================================================
# TRANSFORMER CUSTOMIZADO
# Precisa estar definido aqui para o joblib conseguir desserializar.
# =====================================================================
class PreProcessadorImoveis(BaseEstimator, TransformerMixin):
    """Aplica os tratamentos definidos no EDA do projeto."""

    def fit(self, X, y=None):
        X = X.copy()
        self.lotfront_por_bairro_ = (
            X.groupby('Neighborhood')['LotFrontage'].median()
        )
        self.lotfront_global_ = X['LotFrontage'].median()
        self.electrical_moda_ = X['Electrical'].mode(dropna=True).iloc[0]
        return self

    def transform(self, X):
        X = X.copy()

        for c in COLS_NA_NONE:
            if c in X.columns:
                X[c] = X[c].fillna('None')
        for c in COLS_NA_ZERO:
            if c in X.columns:
                X[c] = X[c].fillna(0)

        X['LotFrontage'] = X.apply(
            lambda r: self.lotfront_por_bairro_.get(
                r['Neighborhood'], self.lotfront_global_
            ) if pd.isna(r['LotFrontage']) else r['LotFrontage'],
            axis=1,
        )
        X['Electrical'] = X['Electrical'].fillna(self.electrical_moda_)

        for c in COLS_ORDINAL_QUAL:
            if c in X.columns:
                X[c] = X[c].map(MAP_QUAL).fillna(0).astype(float)

        X['TotalSF'] = (
            X['TotalBsmtSF'].fillna(0)
            + X['1stFlrSF'].fillna(0)
            + X['2ndFlrSF'].fillna(0)
        )
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']

        return X


# =====================================================================
# FUNÇÃO OBRIGATÓRIA — chamada pelo corretor automático
# =====================================================================
def prever_precos(caminho_arquivo_teste):
    """
    Lê o CSV de teste, aplica todo o pré-processamento via Pipeline
    treinado e retorna as predições de preço em dólares.

    Parâmetros:
        caminho_arquivo_teste (str): caminho do CSV de teste.

    Retorna:
        np.array com as predições em dólares, na mesma ordem das linhas.
    """
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # Remove Id e SalePrice se presentes (não são features)
    if 'Id' in df_teste.columns:
        df_teste = df_teste.drop(columns=['Id'])
    if 'SalePrice' in df_teste.columns:
        df_teste = df_teste.drop(columns=['SalePrice'])

    caminho_modelo = os.path.join(os.path.dirname(__file__), 'modelo.joblib')
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Arquivo '{caminho_modelo}' não encontrado. "
            "Certifique-se de que modelo.joblib está na raiz do repositório."
        )
    pipeline = joblib.load(caminho_modelo)

    # O modelo foi treinado com log1p(SalePrice); revertemos com expm1.
    predicoes_log = pipeline.predict(df_teste)
    predicoes = np.expm1(predicoes_log)
    predicoes = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes


# =====================================================================
# VALIDAÇÃO LOCAL (executado apenas quando rodamos `python pipeline.py`)
# =====================================================================
if __name__ == '__main__':
    arquivo_teste = os.path.join(os.path.dirname(__file__), 'teste_publico.csv')

    print('─' * 50)
    print('  Validação Local do Pipeline')
    print('─' * 50)

    if not os.path.exists(arquivo_teste):
        print(f"[Aviso] '{arquivo_teste}' não encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste)

            print('✅ Pipeline executado com sucesso!')
            print(f'   Total de predições : {len(resultados)}')
            print(f'   Primeiras 5        : {np.round(resultados[:5], 2)}')
            print(f'   Mín / Máx          : ${resultados.min():,.0f} / ${resultados.max():,.0f}')
            print(f'   Média              : ${resultados.mean():,.0f}')

            df_val = pd.read_csv(arquivo_teste)
            if 'SalePrice' in df_val.columns:
                rmsle = np.sqrt(mean_squared_log_error(df_val['SalePrice'], resultados))
                print(f'\n   RMSLE local        : {rmsle:.5f}')
                print(f'   Baseline professor : 0.17543')
                if rmsle < 0.17543:
                    print(f'   ✅ Supera o baseline! ({rmsle:.5f} < 0.17543)')
                else:
                    print(f'   ⚠ Abaixo do baseline.')
            else:
                print('\n   [Nota] SalePrice não está no CSV — RMSLE não calculado.')

        except Exception as e:
            print(f'❌ Erro no pipeline:\n{e}')

    print('─' * 50)
