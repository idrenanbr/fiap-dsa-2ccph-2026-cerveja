"""Testes de integração: executar a partir da pasta do projeto."""
import json
from pathlib import Path
import joblib
import numpy as np
from streamlit.testing.v1 import AppTest
from analysis import ROOT, load_data, FEATURES

df,quality=load_data()
assert len(df)==365 and quality['ausentes']==0
for name in ['Ridge','XGBoost','Rede_profunda']:
    model=joblib.load(ROOT/'artifacts'/(name+'.joblib'))
    p=model.predict(df[FEATURES].iloc[:3])
    assert len(p)==3 and np.isfinite(p).all()
at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
assert not at.exception, str(at.exception)
assert at.metric[0].value=='365'
at.sidebar.selectbox[0].select('Final de semana').run()
assert not at.exception
assert int(at.metric[0].value)==104
at.sidebar.selectbox[1].select('Com chuva').run()
assert not at.exception
assert int(at.metric[0].value)<104
# Intervalo sem dados após filtro; deve exibir aviso, sem exceção.
at.sidebar.date_input[0].set_value((df.Data.min().date(),df.Data.min().date())).run()
assert not at.exception and any('Não há dados nesse recorte' in x.value for x in at.warning)
# Uma data selecionada: app não quebra.
at.sidebar.date_input[0].set_value((df.Data.min().date(),)).run()
assert not at.exception
# Cenário válido para cada modelo.
for name in ['Ridge','XGBoost','Rede profunda']:
    next(x for x in at.selectbox if x.label=='Modelo').select(name)
    at.button[0].click().run()
    assert not at.exception, str(at.exception)
    assert any(m.label.startswith('Consumo estimado') for m in at.metric)
# Mínima maior que máxima deve ser rejeitada.
at.number_input[1].set_value(50.)
at.button[0].click().run()
assert not at.exception and any('temperatura mínima' in x.value for x in at.error)
report={'status':'PASS','testes':['365 dias válidos','3 artefatos carregam e predizem',
    'interface inicia sem exceções','filtro fim de semana: 104 dias',
    'filtro chuva','filtro vazio','data única','previsão com cada modelo','temperaturas inconsistentes rejeitadas']}
(ROOT/'artifacts/test_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
