"""Dados, indicadores e treinamento reproduzível dos checkpoints."""
from pathlib import Path
import hashlib
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
TARGET = 'Consumo de cerveja (litros)'
FEATURES = ['Temperatura Media (C)', 'Temperatura Minima (C)',
            'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Final de Semana']
SOURCE = 'https://www.kaggle.com/datasets/dongeorge/beer-consumption-sao-paulo'

def load_data(path=None):
    path = Path(path or ROOT / 'data/Consumo_cerveja.csv')
    raw = pd.read_csv(path)
    required = ['Data', *FEATURES, TARGET]
    if not set(required).issubset(raw.columns):
        raise ValueError('Colunas obrigatórias ausentes')
    df = raw.dropna(how='all').copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='raise')
    df['Data'] = pd.to_datetime(df['Data'], errors='raise')
    if df[required].isna().any().any():
        raise ValueError('Dados parcialmente ausentes: revisar antes de analisar')
    if df['Data'].duplicated().any() or df.duplicated().any():
        raise ValueError('Datas/registros duplicados')
    if not df['Final de Semana'].isin([0, 1]).all():
        raise ValueError('Indicador de fim de semana inválido')
    if not (df['Final de Semana'].astype(int).values == (df['Data'].dt.day_of_week >= 5).astype(int).values).all():
        raise ValueError('Indicador de fim de semana não corresponde às datas')
    if (df['Precipitacao (mm)'] < 0).any() or (df[TARGET] < 0).any():
        raise ValueError('Chuva ou consumo negativo')
    if not ((df[FEATURES[1]] <= df[FEATURES[0]]) & (df[FEATURES[0]] <= df[FEATURES[2]])).all():
        raise ValueError('Temperaturas inconsistentes')
    df['Final de Semana'] = df['Final de Semana'].astype(int)
    df = df.sort_values('Data').reset_index(drop=True)
    quality = {'linhas_originais': len(raw), 'linhas_vazias_removidas': len(raw)-len(df),
               'linhas_validas': len(df), 'ausentes': int(df[required].isna().sum().sum()),
               'duplicadas': int(df.duplicated().sum()),
               'sha256_csv': hashlib.sha256(path.read_bytes()).hexdigest()}
    return df, quality

def indicators(df):
    if df.empty:
        return {}
    weekend = df.groupby('Final de Semana')[TARGET].mean()
    rain = df.groupby(df['Precipitacao (mm)'].gt(0))[TARGET].mean()
    return {'dias': len(df), 'consumo_medio': float(df[TARGET].mean()),
            'consumo_total_periodo': float(df[TARGET].sum()),
            'media_fim_semana': float(weekend.get(1, np.nan)),
            'media_dia_util': float(weekend.get(0, np.nan)),
            'diferenca_fim_semana_pct': float((weekend.get(1, np.nan)/weekend.get(0, np.nan)-1)*100),
            'media_com_chuva': float(rain.get(True, np.nan)),
            'media_sem_chuva': float(rain.get(False, np.nan)),
            'correlacao_tmax': float(df[FEATURES[2]].corr(df[TARGET])) if len(df)>1 else np.nan}

def score(model, X, y):
    p = model.predict(X)
    return {'MAE': float(mean_absolute_error(y,p)),
            'RMSE': float(np.sqrt(mean_squared_error(y,p))), 'R2': float(r2_score(y,p))}

def train(out=None):
    out = Path(out or ROOT/'artifacts'); out.mkdir(parents=True, exist_ok=True)
    df, quality = load_data()
    # Avaliação prospectiva: passado para treino, depois validação e teste final.
    n1, n2 = int(len(df)*.6), int(len(df)*.8)
    a, b, c = df.iloc[:n1], df.iloc[n1:n2], df.iloc[n2:]
    models = {
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'XGBoost': XGBRegressor(n_estimators=180, max_depth=2, learning_rate=.03,
                               subsample=.9, colsample_bytree=1., random_state=42, n_jobs=1),
        'Rede profunda': TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(64,32,16), activation='relu', solver='adam',
                alpha=.1, max_iter=2500, early_stopping=False, random_state=42)),
            transformer=StandardScaler())}
    rows=[]; warning_log=[]
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model.fit(a[FEATURES],a[TARGET])
        warning_log.extend([{'modelo':name,'aviso':str(w.message)} for w in caught])
        rows.append({'modelo':name, 'fase':'validacao', **score(model,b[FEATURES],b[TARGET])})
    winner = min(rows,key=lambda r:r['RMSE'])['modelo']
    # Modelo é escolhido exclusivamente pela validação, antes de avaliar o teste.
    ab = df.iloc[:n2]
    predictions = pd.DataFrame({'Data':c['Data'], 'observado':c[TARGET]})
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model.fit(ab[FEATURES],ab[TARGET])
        warning_log.extend([{'modelo':name,'aviso':str(w.message)} for w in caught])
        rows.append({'modelo':name,'fase':'teste',**score(model,c[FEATURES],c[TARGET])})
        predictions[name] = model.predict(c[FEATURES])
        joblib.dump(model,out/(name.replace(' ','_')+'.joblib'))
    baseline=DummyRegressor().fit(ab[FEATURES],ab[TARGET])
    rows.append({'modelo':'Baseline média','fase':'teste',**score(baseline,c[FEATURES],c[TARGET])})
    predictions['Baseline média']=baseline.predict(c[FEATURES])
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'metrics.csv', index=False)
    predictions.to_csv(out/'predictions_test.csv',index=False)
    split = {name:{'n':len(x),'inicio':str(x.Data.min().date()),'fim':str(x.Data.max().date())}
             for name,x in [('treino',a),('validacao',b),('teste',c)]}
    metadata={'seed':42,'winner_validation_rmse':winner,'features':FEATURES,'target':TARGET,
              'split':split,'quality':quality,'warnings':warning_log,
              'trained_until':str(ab.Data.max().date()),
              'note':'Artefatos treinados em treino+validação; teste não usado no ajuste ou escolha.'}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics, metadata

if __name__=='__main__':
    metrics, metadata=train(); print(metrics.to_string(index=False)); print(json.dumps(metadata,ensure_ascii=False,indent=2))
