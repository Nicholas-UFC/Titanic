# Projeto Titanic: Previsão de Sobrevivência (Kaggle)

## Visão Geral do Projeto

Este projeto aborda o famoso desafio "Titanic - Machine Learning from Disaster" do Kaggle, onde o objetivo é prever a sobrevivência de passageiros a bordo do RMS Titanic com base em um conjunto de dados histórico. Através deste notebook, foi explorado um pipeline completo de Ciência de Dados, desde a análise exploratória de dados (EDA) e engenharia de features, até a construção, otimização e avaliação de múltiplos modelos de Machine Learning, culminando na criação de um modelo de ensemble para submissão no Kaggle.

## Problema

O naufrágio do RMS Titanic é uma das tragédias mais conhecidas da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou após colidir com um iceberg, resultando na morte de 1502 das 2224 pessoas a bordo. Embora o fator sorte tenha desempenhado um papel, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros. Este projeto visa construir um modelo preditivo para determinar se um passageiro sobreviveu ou não.

## Dataset

Os dados utilizados são fornecidos pelo Kaggle para a competição do Titanic e incluem:

* `train.csv`: Contém os dados de treinamento, incluindo a variável alvo (`Survived`).
* `test.csv`: Contém os dados de teste, sem a variável alvo, para os quais as previsões devem ser feitas.
* `gender_submission.csv`: Um exemplo de arquivo de submissão no formato correto.
* `Titanic.ipynb`: O notebook Jupyter que contém toda a análise e o código do projeto.
* `submission_titanic_rf_otimizado.csv`: O arquivo de submissão gerado por este projeto.

### Dicionário de Dados (Features)

* `PassengerId`: ID único do passageiro.
* `Survived`: Sobrevivente (0 = Não, 1 = Sim) - Variável Alvo.
* `Pclass`: Classe do ingresso (1 = 1ª, 2 = 2ª, 3 = 3ª).
* `Name`: Nome do passageiro.
* `Sex`: Sexo (male/female).
* `Age`: Idade em anos.
* `SibSp`: Número de irmãos/cônjuges a bordo.
* `Parch`: Número de pais/filhos a bordo.
* `Ticket`: Número do ticket.
* `Fare`: Tarifa do passageiro.
* `Cabin`: Número da cabine.
* `Embarked`: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton).

## Metodologia e Abordagem

O projeto seguiu as etapas padrão de um pipeline de Machine Learning:

1.  **Carregamento e Exploração Inicial de Dados:**
    * Compreensão da estrutura dos datasets `train.csv` e `test.csv`.
    * Identificação de tipos de dados e valores ausentes.

2.  **Limpeza e Pré-processamento de Dados:**
    * **Tratamento de Valores Ausentes:**
        * Coluna `Cabin`: Removida devido ao alto percentual de valores ausentes.
        * Coluna `Age`: Preenchida com a mediana das idades para manter a robustez a outliers.
        * Coluna `Embarked`: Preenchida com a moda (porto mais frequente).
        * Coluna `Fare` (no `test.csv`): Preenchida com a mediana da tarifa do conjunto de treino.
    * **Transformação de Features:**
        * `Fare_log`: Aplicação de transformação logarítmica na tarifa para reduzir assimetria e impacto de outliers.
        * `FamilySize_log`: Aplicação de transformação logarítmica no tamanho da família para aprimorar a distribuição.

3.  **Engenharia de Features:**
    * Criação da feature `FamilySize`: Combinou `SibSp` e `Parch` (irmãos/cônjuges + pais/filhos) adicionando 1 (para incluir o próprio passageiro), revelando insights sobre o impacto do tamanho da família na sobrevivência.

4.  **Análise Exploratória de Dados (EDA):**
    * Análise detalhada do impacto de `Sex`, `Pclass`, `Age`, `Fare` e `FamilySize` na variável `Survived`.
    * Utilização de visualizações (histogramas, KDE plots, gráficos de barras) para identificar padrões e correlações.
    * Insights chave: Mulheres e crianças tinham maior taxa de sobrevivência; a classe e a tarifa pagas impactaram fortemente as chances de sobreviver; famílias de tamanho médio (2-4) tinham maior chance, enquanto muito grandes ou muito pequenas (sozinhos) tinham menos.

5.  **Preparação de Dados para Modelagem:**
    * **One-Hot Encoding:** Variáveis categóricas (`Sex`, `Embarked`) foram convertidas para um formato numérico binário.
    * **Separação X e Y:** As features (X) e a variável alvo (Y - `Survived`) foram definidas.
    * **Divisão Treino/Teste:** Os dados foram divididos em conjuntos de treino (80%) e teste (20%) para avaliação imparcial do modelo.
    * **Escalonamento de Features:** As features numéricas (`Pclass`, `Age`, `Fare_log`, `FamilySize_log`) foram padronizadas usando `StandardScaler` para garantir que modelos baseados em distância ou gradiente funcionassem corretamente.
    * **Alinhamento de Colunas:** Garantiu-se que o conjunto de teste tivesse exatamente as mesmas colunas na mesma ordem que o conjunto de treino.

6.  **Modelagem e Otimização:**
    * Foram explorados e otimizados diversos algoritmos de Machine Learning usando **`GridSearchCV`** (com validação cruzada 5-fold) para encontrar os melhores hiperparâmetros:
        * **Regressão Logística**
        * **Random Forest Classifier**
        * **XGBoost Classifier**
        * **K-Nearest Neighbors (KNN)**
        * **Support Vector Machine (SVM)**
    * O `GridSearchCV` foi crucial para refinar o desempenho de cada modelo.

7.  **Ensemble Learning (Voting Classifier):**
    * Foi construído e avaliado um `VotingClassifier` (com votação suave) combinando os modelos otimizados de Random Forest, SVM e KNN, visando uma performance preditiva ainda mais robusta e superior.

8.  **Avaliação e Interpretação do Modelo:**
    * O desempenho dos modelos foi avaliado usando **Acurácia**, **Matriz de Confusão** e **Relatório de Classificação** (Precision, Recall, F1-Score).
    * A **Importância das Features** foi extraída do melhor modelo (Random Forest) para entender quais fatores mais influenciaram suas previsões.
    * A **Validação Cruzada** foi aplicada ao modelo final para uma estimativa mais robusta de sua capacidade de generalização.

## Resultados

O **Voting Classifier** demonstrou ser o modelo com melhor desempenho no conjunto de teste, com uma acurácia de **83.24%**.

| Modelo                 | Acurácia no Conjunto de Teste (20%) | Acurácia Média CV (no Treino - 5 Folds) |
| :--------------------- | :---------------------------------- | :-------------------------------------- |
| Regressão Logística (Base) | 0.8045                              | N/A                                     |
| Regressão Logística (Otimizada) | 0.7877                              | 0.8061                                  |
| Random Forest (Base)   | 0.7989                              | N/A                                     |
| **Random Forest (Otimizado)** | **0.8212** | **0.8339** |
| XGBoost (Base)         | 0.7933                              | N/A                                     |
| XGBoost (Otimizado)    | 0.7821                              | 0.8329                                  |
| KNN (Base)             | 0.7877                              | N/A                                     |
| KNN (Otimizado)        | 0.8156                              | 0.8259                                  |
| SVM (Base)             | 0.8045                              | N/A                                     |
| **SVM (Otimizado)** | **0.8212** | **0.8259** |
| **Voting Classifier** | **0.8324** | 0.8182                                  |

### Insights Chave dos Fatores de Sobrevivência (Validados pelo Modelo)

As features mais importantes para a previsão de sobrevivência, de acordo com o modelo Random Forest, foram:

1.  **Tarifa Paga (`Fare_log`)**: Indica o status socioeconômico e a localização no navio.
2.  **Idade (`Age`)**: Crianças tinham maior prioridade.
3.  **Sexo (`Sex_male` / `Sex_female`)**: Mulheres tinham uma chance drasticamente maior de sobreviver.
4.  **Classe do Passageiro (`Pclass`)**: Fortemente correlacionada à tarifa e localização.
5.  **Tamanho da Família (`FamilySize_log`)**: Famílias de tamanho intermediário tinham vantagem.

## Como Reproduzir o Projeto

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
    cd SEU_REPOSITORIO](https://github.com/Nicholas-UFC/Titanic.git)
    ```
2.  **Baixe os Dados:** Faça o download dos arquivos `train.csv` e `test.csv` (e `gender_submission.csv` opcionalmente) da competição do Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data). Coloque-os na mesma pasta do `Titanic.ipynb`.
3.  **Instale as Dependências:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```
4.  **Execute o Notebook Jupyter:** Abra o arquivo `Titanic.ipynb` em um ambiente como Jupyter Notebook ou VS Code e execute todas as células em sequência. O arquivo de submissão (`submission_titanic_rf_otimizado.csv`) será gerado na mesma pasta.

## Tecnologias Utilizadas

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* XGBoost

## Autor

Bryan Nicholas

---
