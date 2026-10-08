# Tech Challenge — Fase 2 | POSTECH Data Analytics

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma |  2DTATBB |
| Grupo | Grupo 5 |
| Data de entrega | <!-- PREENCHER: DD/MM/AAAA --> |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Stephane Abreu de Oliveira | RM377745 | stephaneeao@gmail.com |
| Bruno Alves de Moura Soares | RM377837 | bruno_nfsu@hotmail.com |
| Dalura Lummy Dionisio de Moraes Fernandes | RM377761 | daluralummy@bb.com.br |
| Felipe Roberto Luvizotti | RM377822 | luvizotti@bb.com.br |
| Melissa Santiago dos Santos Cruz | RM377778 | melzinhacruz@gmail.com |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/stephaneeao-cmd/tech-challenge-fase2-grupo5 |
| Vídeo executivo (≤ 5 min) | [Vídeo executivo (Google Drive)](https://drive.google.com/file/d/13yjmH-mwJrJaJ2bZnhXqrfZEkiqLUeQG/view?usp=sharing) |
| Apresentação | [PDF da apresentação (Google Drive)](https://drive.google.com/file/d/1Mms1_4V8ao0HnnCu8D5rmf1wECcEVYVJ/view?usp=sharing) |

> ⚠️ Repositório privado ou inacessível inviabiliza a avaliação da entrega.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema

A concessão de cartões de crédito exige avaliar o risco de inadimplência a partir dos dados declarados pelos clientes no momento da solicitação. A motivação para o uso de Machine Learning é analisar se informações puramente cadastrais e demográficas conseguem antecipar atrasos graves de pagamento. No entanto, o estudo revelou uma inconsistência crítica nos dados: identificaram-se 269 perfis cadastrais com exatamente os mesmos atributos associados a alvos divergentes (1.574 clientes), demonstrando que apenas os dados cadastrais são insuficientes para separar o comportamento financeiro desses clientes.

### Variável alvo

* **Variável alvo:** `TARGET`.
* **Definição e limiar adotado:** O alvo foi construído a partir da coluna `STATUS` da base de histórico de crédito.
Definiu-se `TARGET = 1` (atraso grave) para proponentes com registo de atraso igual ou superior a 60 dias (valores `2`, `3`, `4` ou `5` em qualquer mês). Definiu-se `TARGET = 0` para clientes sem atrasos de 60 dias ou mais no histórico analisado (ressaltando que essa classe ainda pode conter registos com atrasos inferiores a 60 dias).
* **Distribuição das classes:** Forte desbalanceamento de classes. Na exploração isolada do histórico, 667 de 45.985 clientes (1,45%) tiveram atraso grave. Após a junção interna (*merge*) com a base cadastral, resultaram 36.457 clientes únicos, dos quais 35.841 pertencem à classe `0` (98,31%) e apenas 616 à classe `1` (1,69%).

### Dataset
Campo | Valor |
|---|---|
| Fonte | Base fornecida pela FIAP ([Google Drive](https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing)), originalmente do Kaggle — Credit Card Approval Prediction |
| Linhas × colunas | Aplicações: 438.557 × 18 \| Histórico de crédito: 1.048.575 × 3 |
| Período / versão | Conjunto público consolidado|
| Licença de uso | Domínio Público (CC0) |

Descrição das variáveis:

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | Inteiro | Identificador do proponente (removido das features de treino) |
| `GENERO` | Categórico | Género informado no cadastro (`CODE_GENDER`) |
| `POSSUI_CARRO` | Categórico | Indicador de posse de veículo (`FLAG_OWN_CAR`) |
| `POSSUI_IMOVEL` | Categórico | Indicador de posse de imóvel próprio (`FLAG_OWN_REALTY`) |
| `NUM_FILHOS` | Numérico | Número de filhos do solicitante (`CNT_CHILDREN`) |
| `RENDA_ANUAL` | Numérico | Renda anual total informada (`AMT_INCOME_TOTAL`) |
| `TIPO_RENDA` | Categórico | Categoria da fonte de rendimentos (`NAME_INCOME_TYPE`) |
| `ESCOLARIDADE` | Categórico | Nível de escolaridade concluído (`NAME_EDUCATION_TYPE`)|
| `ESTADO_CIVIL` | Categórico | Estado civil informado (`NAME_FAMILY_STATUS`) |
| `TIPO_MORADIA` | Categórico | Tipo de habitação/moradia (`NAME_HOUSING_TYPE`) |
| `IDADE_ANOS` | Numérico | Idade do cliente convertida em anos a partir de `DAYS_BIRTH`|
| `TEMPO_TRABALHO_ANOS` | Numérico | Tempo de trabalho em anos derivado de `DAYS_EMPLOYED` (com código 365243 convertido em nulo) |
| `TELEFONE_TRABALHO` | Binário | Indicador de registo de telefone profissional (`FLAG_WORK_PHONE`)|
| `TELEFONE` | Binário | Indicador de telefone residencial/fixo (`FLAG_PHONE`) |
| `POSSUI_EMAIL` | Binário | Indicador de endereço de e-mail (`FLAG_EMAIL`) |
| `OCUPACAO` | Categórico | Ocupação do solicitante (valores nulos e raros agrupados em `Outros`) |
| `TAMANHO_FAMILIA` | Numérico | Quantidade de membros na família (`CNT_FAM_MEMBERS`) |
| `MESES_HISTORICO` | Numérico | Mês relativo no histórico (0 = atual; negativos = meses passados) |
| `STATUS` | Categórico | Situação do pagamento mensal (`0` a `5`, `C`, `X`) |
| `TARGET` | Binário | Alvo preditivo binarizado (`1` = atraso ≥ 60 dias, `0` = sem atraso grave) |

*(Nota: a variável original `POSSUI_CELULAR` / `FLAG_MOBIL` foi excluída durante o pré-processamento por ser uma constante com variância zero).*

---

## 4. Como reproduzir

```bash
git clone https://github.com/stephaneeao-cmd/tech-challenge-fase2-grupo5.git
cd tech-challenge-fase2-grupo5

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque os arquivos `application_record.csv` e `credit_record.csv` em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:


| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória, renomeação de variáveis para português, verificação de dados faltantes, detecção de outliers e análise do desbalanceamento do target. |
| 2 | `notebooks/02_preprocessamento.ipynb` | Remoção de IDs duplicados, binarização do target (status 2 a 5 como atraso grave), imputação de ocupação, conversão de idade e tempo de trabalho para anos e exclusão da coluna constante. |
| 3 | `notebooks/03_modelagem.ipynb` | Agrupamento por perfis cadastrais idênticos, divisão dos dados com `StratifiedGroupKFold`, configuração dos pipelines e comparação de 5 modelos (Regressão Logística, Árvore de Decisão, KNN, SVM Linear e Random Forest). |
| 4 | `notebooks/04_avaliacao.ipynb` | Avaliação do modelo campeão (Árvore de Decisão) no conjunto de teste com perfis inéditos, cálculo de métricas, matriz de confusão, curva ROC e análise de importância das variáveis. |

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

---

## 5. Resultados

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Árvore de Decisão (Modelo Selecionado - Teste)** | 0.945 | 0.037 | 0.089 | 0.052 | 0.526 |
| Árvore de Decisão (Validação Cruzada por Perfil) | — | — | 0.172 | 0.089 | — |
| Regressão Logística (Validação Cruzada por Perfil) | — | — | 0.465 | 0.038 | — |
| SVM Linear (Validação Cruzada por Perfil) | — | — | 0.462 | 0.038 | — |
| Random Forest (Validação Cruzada por Perfil) | — | — | 0.014 | 0.027 | — |
| KNN (Validação Cruzada por Perfil) | — | — | 0.008 | 0.013 | — |

**Modelo escolhido:** **Árvore de Decisão (`DecisionTreeClassifier`)** — Foi selecionada por alcançar o maior F1-score médio (0,089) na validação cruzada agrupada por perfil (`StratifiedGroupKFold`), superando os demais modelos na capacidade de equilibrar precisão e identificação de inadimplentes em perfis não vistos.

**Métricas priorizadas:** Foram priorizados o **F1-Score** e o **Recall** voltados à classe minoritária (`TARGET = 1`). Como a base apresenta um desbalanceamento severo (apenas 1,69% de clientes com atraso grave), a **Acurácia (0,945)** é uma métrica enganosa, pois um modelo ingênuo que aprove todos os cadastros já atingiria mais de 98% de acerto. No contexto bancário, o custo de um Falso Negativo (conceder crédito a quem terá atraso grave) é substancialmente superior ao de um Falso Positivo (recusar ou analisar manualmente um bom pagador); contudo, modelos como Regressão Logística e SVM geraram um volume insustentável de falsos alertas, motivo pelo qual o F1 agrupado foi o critério decisivo.


---



## 6. Principais conclusões

1. **Variáveis mais determinantes:** De acordo com a importância de atributos calculada na Árvore de Decisão, as variáveis mais relevantes foram `IDADE_ANOS` (24,1%), `TEMPO_TRABALHO_ANOS` (20,0%) e `RENDA_ANUAL` (15,4%), demonstrando que o ciclo de vida, a estabilidade profissional e o nível de renda concentram a maior parte do poder explicativo do modelo.
2. **Insuficiência dos dados puramente cadastrais:** A auditoria revelou a existência de 269 perfis cadastrais idênticos associados a comportamentos de pagamento opostos (1.574 proponentes). Isso evidencia que atributos estáticos (como moradia, escolaridade e posse de bens) não conseguem, isoladamente, explicar o risco de inadimplência.
3. **Alto índice de falsos alertas e baixa precisão:** Na avaliação com perfis não vistos no conjunto de teste, o modelo identificou apenas 11 dos 123 clientes com atraso grave (recall de 8,9%) e gerou 288 falsos positivos. Isso resultou em uma precisão de apenas 3,7%, o que na prática bancária significa bloquear ou exigir revisão manual excessiva de clientes solventes.
4. **Dificuldade de generalização e inviabilidade de automação integral:** Com um AUC-ROC de 0,526 (muito próximo da escolha aleatória de 0,50), o modelo apresenta baixa capacidade de discriminar novos clientes. Portanto, ele não deve ser utilizado para decisões automáticas de concessão de crédito sem o suporte de análise humana ou de fontes de dados complementares.

### Limitações e próximos passos

* **Limitações:**
  * **Conflito de perfis idênticos:** Clientes com os mesmos atributos cadastrais apresentaram alvos distintos na base histórica, impondo um teto estrutural à capacidade preditiva dos algoritmos com as variáveis atuais.
  * **Severidade da validação por perfil:** A divisão agrupada (`StratifiedGroupKFold`) mede o desempenho estritamente sobre combinações nunca vistas, penalizando métricas como recall e precisão quando comparadas à validação aleatória simples.
  * **Definição simplificada do alvo:** A variável de atraso grave (status ≥ 60 dias) resume o histórico observado, mas não cobre todo o espectro de risco de crédito, podendo classificar atrasos menores (como 30 dias) como adimplentes.

* **Próximos passos:**
  * **Incorporação de novos dados:** Integrar variáveis dinâmicas de comportamento financeiro, histórico de pagamento de faturas, consultas a birôs de crédito externos e dados de Open Finance.
  * **Calibração de regras de negócio:** Alinhar com a área de crédito o custo financeiro tolerável de falsos positivos (fricção e perda de clientes bons) versus falsos negativos (perdas por inadimplência) para ajuste do limiar de corte (*threshold*).
  * **Testes em outras safras:** Realizar testes fora do tempo (*out-of-time*) para avaliar a robustez temporal do modelo e checar eventuais disparidades em subgrupos demográficos.


---

## 7. Estrutura do repositório

```
.
├── data/              dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/         análise em ordem numerada (01_eda → 04_avaliacao)
├── results/
│   ├── figures/       gráficos gerados pelos notebooks
│   ├── metrics/       métricas de validação e teste (CSV)
│   └── models/        modelos treinados
├── docs/              apresentação executiva (PDF disponível no link da seção 2)
├── submissao/         entrega.json e script que gera o PDF de submissão
└── requirements.txt   dependências do projeto
```

---

## 8. Tecnologias

- Python 3.12.4
- pandas 3.0.6 — manipulação dos dados
- numpy 2.4.3 — cálculos numéricos
- scikit-learn 1.9.1 — pré-processamento, modelos e métricas
- matplotlib 3.11.2 / seaborn 0.13.2 — visualizações
- joblib 1.6.0 — salvar os modelos treinados
- jupyter 1.1.1 — execução dos notebooks
- reportlab 5.0.1 — geração do PDF de submissão
