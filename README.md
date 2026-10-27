# Consumo de cerveja em São Paulo

Projeto da disciplina Data Science & Analytics — FIAP 2CCPH-2026, segundo semestre de 2026.
Integra as três etapas: problema e indicadores (CP1/CP4), modelagem (CP2/CP5) e dashboard (CP3/CP6).

**[Abrir a aplicação](https://fiap-dsa-2ccph-2026-cerveja.streamlit.app/)** · **[Testes Linux aprovados](https://github.com/idrenanbr/fiap-dsa-2ccph-2026-cerveja/actions/runs/40502139953)**

Acesso sem autenticação verificado em 27/10/2026.

## O que a aplicação faz

- Explora 365 observações diárias de 2015, com filtros por período, tipo de dia e chuva.
- Calcula KPIs e mostra gráficos estáticos e interativos.
- Compara Ridge, XGBoost e rede neural profunda com três camadas ocultas.
- Simula consumo com pipelines treinados, sem treinamento ao abrir a página.
- Explica os dados, as métricas, as limitações da análise.

## Executar localmente

Use Python 3.12. A partir desta pasta:

```text
python -m venv .venv
```

Ative o ambiente (`.venv\Scripts\activate` no Windows; `source .venv/bin/activate` no Linux/macOS) e execute:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Reproduzir os resultados

```text
python analysis.py
python test_app.py
python executar_notebooks.py
```

`analysis.py` recalcula os modelos e arquivos em `artifacts/`. Os notebooks já incluem saídas de uma execução real.
Para reexecutá-los, é necessário um kernel `python3` com as dependências instaladas. O teste de interface usa Streamlit AppTest.
A automação em `.github/workflows/validar.yml` executa esses testes e os três notebooks em Linux/Python 3.12.

## Dados e método

Fonte de distribuição: [Beer Consumption – São Paulo, Kaggle](https://www.kaggle.com/datasets/dongeorge/beer-consumption-sao-paulo).
Consulta documental em 27/10/2026; o arquivo foi recuperado dos materiais anteriores. A data original do download não está registrada.
O conjunto tem sete colunas: data; temperaturas média, mínima e máxima; precipitação; fim de semana; consumo.
O CSV original tem 941 linhas, das quais 576 totalmente vazias. A análise mantém 365 dias válidos.
O hash do arquivo está em `artifacts/metadata.json`.

A divisão temporal reserva 219 dias para treino, 73 para validação e 73 para teste.
O modelo padrão é escolhido pelo RMSE da validação. O ajuste final usa treino e validação; o teste permanece separado.
No teste final, o Ridge selecionado obteve MAE 2,215 e RMSE 2,619, contra RMSE 4,101 da baseline pela média.
Consulte a tabela completa em `artifacts/metrics.csv`.

Os valores preservam a escala e o rótulo de unidade do CSV. Não foi aplicada conversão arbitrária por mil.
Os dados não identificam estudantes, idade, horários ou feriados. Não sustentam inferência causal nem generalização para toda São Paulo ou para 2026.
Para demanda futura, seria preciso usar previsões meteorológicas e medir também esse erro.

## Arquivos

| Arquivo | Finalidade |
|---|---|
| `CP1_corrigido.ipynb` | Problema, EDA, três perguntas, KPIs calculados e respostas |
| `CP2_corrigido.ipynb` | Preparação, modelos, avaliação e conexão com indicadores |
| `CP3_corrigido.ipynb` | Implementação e teste executado do Streamlit |
| `analysis.py` | Leitura validada, indicadores, treinamento e persistência |
| `app.py` | Dashboard Streamlit |
| `test_app.py` | Testes de interface e carregamento dos três modelos |
| `artifacts/` | Pipelines treinados, métricas, previsões e evidências de execução |
| `data/` | Dataset usado nas três etapas |

Os enunciados originais, a auditoria da entrega anterior e a mensagem ao professor ficam no pacote privado de revisão, fora deste repositório público.
Nenhum envio no Teams ou ao professor é feito por este projeto.

## Identificação da entrega

Os nomes e RMs completos estão nos notebooks do pacote privado de entrega. A versão pública não divulga esses identificadores.
