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
st.caption('Data Science & Analytics · FIAP 2CCPH-2026 · CP3/CP6')
st.markdown('Dados diários de 2015, disponíveis no [Kaggle]('+SOURCE+'). Fonte consultada em 27/10/2026.')
st.info('Explore como o consumo variou com o clima e os dias da semana. Os resultados se referem a esta base de 2015; não representam toda a população de São Paulo.')

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
explore,predict,methods=st.tabs(['Dados e indicadores','Previsões','Sobre os dados'])
with explore:
    if filtered.empty:
        st.warning('Não há dados nesse recorte. Tente ampliar o período ou mudar os filtros.')
    else:
        k=indicators(filtered); cols=st.columns(3)
        cols[0].metric('Dias observados',k['dias'])
        cols[1].metric('Consumo médio · unidade do CSV',f"{k['consumo_medio']:.3f}")
        cols[2].metric('Variação no fim de semana (%)',f"{k['diferenca_fim_semana_pct']:.1f}" if np.isfinite(k['diferenca_fim_semana_pct']) else 'Não disponível')
        st.caption('A variação compara o consumo médio dos fins de semana com o dos dias úteis. Para calculá-la, o recorte precisa conter os dois tipos de dia.')
        st.subheader('Consumo ao longo do período')
        st.line_chart(filtered.set_index('Data')[[TARGET]],x_label='Data',y_label='Unidade do CSV')
        st.subheader('Temperatura máxima e consumo')
        fig,ax=plt.subplots(figsize=(8,3)); ax.scatter(filtered[FEATURES[2]],filtered[TARGET],alpha=.6)
        ax.set(xlabel='Temperatura máxima (°C)',ylabel='Consumo (unidade do CSV)'); st.pyplot(fig); plt.close(fig)
        st.subheader('Outros indicadores do recorte')
        st.dataframe(pd.DataFrame({'indicador':list(k),'valor':list(k.values())}),hide_index=True)
        st.dataframe(filtered,hide_index=True)
        st.download_button('Baixar dados filtrados',filtered.to_csv(index=False).encode('utf-8'),'dados_filtrados.csv','text/csv')
with predict:
    st.subheader('Como os modelos se saíram')
    st.dataframe(metrics,hide_index=True)
    st.caption('MAE e RMSE mostram o tamanho dos erros: quanto menores, melhor. No R², valores maiores indicam melhor desempenho. Esta comparação usa sempre o mesmo período de teste, independentemente dos filtros da aba de dados.')
    st.write('Modelo escolhido na validação:',meta['winner_validation_rmse'])
    st.dataframe(pd.read_csv(ROOT/'artifacts/predictions_test.csv'),hide_index=True)
    st.subheader('Estime o consumo de um dia')
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
            if outside: st.warning('Os valores a seguir estão fora da faixa observada na base: '+', '.join(outside)+'. A previsão fica menos confiável nesse caso.')
            p=float(model(chosen).predict(row)[0])
            st.metric('Consumo estimado · unidade do CSV',f'{p:.3f}')
            if p<0: st.warning('O modelo retornou um valor negativo, que não faz sentido para consumo. Não use este resultado para planejar estoque.')
            st.caption('Este é um valor estimado, sem intervalo de previsão. Para planejar dias futuros, também seria preciso considerar o erro da previsão do tempo.')
with methods:
    st.write('Os números mantêm a escala do arquivo original, cuja coluna está rotulada como litros. A escala física precisa ser confirmada com a fonte antes de usar os resultados na operação de um estabelecimento.')
    st.write('O objetivo é entender a variação do consumo e avaliar se clima e calendário ajudam a estimar a demanda diária.')
    with st.expander('Qualidade dos dados e detalhes do treinamento'):
        st.json(quality)
        st.json(meta)
    st.markdown('Foram comparados três modelos: Ridge, XGBoost e uma rede neural com três camadas ocultas de 64, 32 e 16 neurônios. A escolha do modelo padrão foi feita na validação, antes de avaliar o teste.')
    st.markdown('A base cobre apenas um ano. Ela não informa a idade ou o perfil dos consumidores, os horários de consumo nem quais dias eram feriados. As relações encontradas são associações, não demonstrações de causa e efeito.')
    st.caption('Os nomes e RMs do grupo estão nos notebooks entregues para revisão.')
