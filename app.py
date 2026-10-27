from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from analysis import ROOT, TARGET, FEATURES, SOURCE, load_data, indicators

st.set_page_config(page_title='Consumo de cerveja | São Paulo',layout='wide')
st.title('Consumo de cerveja em São Paulo')
st.caption('FIAP · 2CCPH-2026 · CP3/CP6 · análise histórica e simulação de demanda')
st.markdown('Fonte: [Beer Consumption – São Paulo]('+SOURCE+'). Dados de 2015; consulta documental em 27/10/2026.')
st.info('Amostra histórica: associações não demonstram causalidade. Os resultados não representam toda a população paulista. A unidade abaixo segue o rótulo do CSV; a escala física deve ser confirmada com a fonte antes de uso operacional.')

@st.cache_data
def data(): return load_data()
@st.cache_resource
def model(name): return joblib.load(ROOT/'artifacts'/(name.replace(' ','_')+'.joblib'))

df,quality=data()
meta=json.loads((ROOT/'artifacts/metadata.json').read_text(encoding='utf-8'))
metrics=pd.read_csv(ROOT/'artifacts/metrics.csv')
st.sidebar.header('Explorar dados')
period=st.sidebar.date_input('Período',value=(df.Data.min().date(),df.Data.max().date()),
                            min_value=df.Data.min().date(),max_value=df.Data.max().date())
day=st.sidebar.selectbox('Tipo de dia',['Todos','Dia útil','Final de semana'])
rain=st.sidebar.selectbox('Chuva',['Todos','Com chuva','Sem chuva'])
filtered=df.copy()
if len(period)==2:
    filtered=filtered[filtered.Data.between(pd.Timestamp(period[0]),pd.Timestamp(period[1]))]
else:
    st.sidebar.caption('Selecione também a data final.')
if day!='Todos': filtered=filtered[filtered['Final de Semana'].eq(int(day=='Final de semana'))]
if rain!='Todos': filtered=filtered[filtered['Precipitacao (mm)'].gt(0).eq(rain=='Com chuva')]
explore,predict,methods=st.tabs(['Indicadores e exploração','Modelos e simulação','Método'])
with explore:
    if filtered.empty:
        st.warning('Nenhum registro para esses filtros. Amplie a seleção.')
    else:
        k=indicators(filtered); cols=st.columns(3)
        cols[0].metric('Dias observados',k['dias'])
        cols[1].metric('Consumo médio · unidade do CSV',f"{k['consumo_medio']:.3f}")
        cols[2].metric('Diferença fim de semana (%)',f"{k['diferenca_fim_semana_pct']:.1f}" if np.isfinite(k['diferenca_fim_semana_pct']) else 'Não disponível')
        st.caption('Diferença = (média fim de semana / média dia útil − 1) × 100. Exige os dois grupos no filtro.')
        st.subheader('Consumo diário · gráfico interativo')
        st.line_chart(filtered.set_index('Data')[[TARGET]],x_label='Data',y_label='Unidade do CSV')
        st.subheader('Temperatura máxima e consumo · gráfico estático')
        fig,ax=plt.subplots(figsize=(8,3)); ax.scatter(filtered[FEATURES[2]],filtered[TARGET],alpha=.6)
        ax.set(xlabel='Temperatura máxima (°C)',ylabel='Consumo (unidade do CSV)'); st.pyplot(fig); plt.close(fig)
        st.subheader('Indicadores por condição')
        st.dataframe(pd.DataFrame({'indicador':list(k),'valor':list(k.values())}),hide_index=True)
        st.dataframe(filtered,hide_index=True)
        st.download_button('Baixar dados filtrados',filtered.to_csv(index=False).encode('utf-8'),'dados_filtrados.csv','text/csv')
with predict:
    st.subheader('Comparação em dados separados no tempo')
    st.dataframe(metrics,hide_index=True)
    st.caption('MAE e RMSE: menores são melhores. R²: maior é melhor, podendo ser negativo. Unidade dos erros = unidade do consumo no CSV. Os filtros exploratórios não retreinam nem alteram a avaliação.')
    st.write('Modelo selecionado por RMSE de validação:',meta['winner_validation_rmse'])
    st.dataframe(pd.read_csv(ROOT/'artifacts/predictions_test.csv'),hide_index=True)
    st.subheader('Simular cenário diário')
    chosen=st.selectbox('Modelo',['Ridge','XGBoost','Rede profunda'],index=['Ridge','XGBoost','Rede profunda'].index(meta['winner_validation_rmse']))
    with st.form('scenario'):
        inputs={}
        for col in FEATURES[:-1]:
            inputs[col]=st.number_input(col,value=float(df[col].median()),min_value=0. if col=='Precipitacao (mm)' else -10.,max_value=300. if col=='Precipitacao (mm)' else 55.)
        inputs['Final de Semana']=int(st.checkbox('Final de semana'))
        submitted=st.form_submit_button('Estimar consumo')
    if submitted:
        if not inputs[FEATURES[1]]<=inputs[FEATURES[0]]<=inputs[FEATURES[2]]:
            st.error('A temperatura mínima deve ser menor ou igual à média, e a média menor ou igual à máxima.')
        else:
            row=pd.DataFrame([inputs],columns=FEATURES)
            outside=[c for c in FEATURES if inputs[c]<df[c].min() or inputs[c]>df[c].max()]
            if outside: st.warning('Cenário fora da faixa histórica: '+', '.join(outside)+'. Extrapolação pouco confiável.')
            p=float(model(chosen).predict(row)[0])
            st.metric('Consumo estimado · unidade do CSV',f'{p:.3f}')
            if p<0: st.warning('Previsão negativa fisicamente inválida; este cenário não deve orientar decisões.')
            st.caption('Estimativa pontual, sem intervalo de confiança. Para antecipar demanda, use previsões meteorológicas: a base contém clima observado, e o erro dessas previsões não foi medido aqui.')
with methods:
    st.write('Objetivo: apoiar planejamento de estoque na amostra de São Paulo, relacionando clima e calendário à demanda diária.')
    st.json(quality); st.json(meta)
    st.markdown('Ridge: referência linear. XGBoost: boosting de árvores. Rede profunda: MLP com três camadas ocultas (64, 32 e 16 neurônios ReLU); features e alvo padronizados apenas com dados de ajuste.')
    st.markdown('Limites: um ano de dados, sem horários, feriados rotulados, perfil individual ou estabelecimentos identificados. Não é possível concluir comportamento de estudantes, jovens ou causalidade.')
    st.caption('A identificação completa do grupo consta nos notebooks do pacote de entrega da disciplina.')
