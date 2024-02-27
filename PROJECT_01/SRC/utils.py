import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# funções
def generate_feature_engineering(dataframe):
    data = dataframe.copy()
    
    # age
    data['AGE_YEARS'] = -data['DAYS_BIRTH'] / 365

    # percentuals
    data['DAYS_EMPLOYED_PERC'] = (data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']).replace(np.inf, 0)
    data['INCOME_CREDIT_PERC'] = (data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']).replace(np.inf, 0)
    data['INCOME_PER_PERSON'] = (data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']).replace(np.inf, 0)
    data['ANNUITY_INCOME_PERC'] = (data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']).replace(np.inf, 0)
    data['PAYMENT_RATE'] = (data['AMT_ANNUITY'] / data['AMT_CREDIT']).replace(np.inf, 0)

    # credit index
    data['CREDIT_TO_GOODS_RATIO'] = (data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']).replace(np.inf, 0)
        
    # revenue index
    data['INCOME_TO_EMPLOYED_RATIO'] = (data['AMT_INCOME_TOTAL'] / data['DAYS_EMPLOYED']).replace(np.inf, 0)
    data['INCOME_TO_BIRTH_RATIO'] = (data['AMT_INCOME_TOTAL'] / data['DAYS_BIRTH']).replace(np.inf, 0)
        
    # time fractions
    data['ID_TO_BIRTH_RATIO'] = (data['DAYS_ID_PUBLISH'] / data['DAYS_BIRTH']).replace(np.inf, 0)
    data['CAR_TO_BIRTH_RATIO'] = (data['OWN_CAR_AGE'] / data['DAYS_BIRTH']).replace(np.inf, 0)
    data['CAR_TO_EMPLOYED_RATIO'] = (data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']).replace(np.inf, 0)
    data['PHONE_TO_BIRTH_RATIO'] = (data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']).replace(np.inf, 0)
    
    return data


# função para devolver alguns detalhes de cada coluna
def generate_metadata(dataframe, target):
    # columns analysis
    columns_descriptions = []

    for column in dataframe.columns:
        if column == target:
            variable = 'target'
        else:
            variable = 'explicable'

        column_description = (column, variable, dataframe[column].nunique(), str(list(dataframe[column].unique())), dataframe[column].isna().sum(), round(dataframe[column].isna().sum() / dataframe.shape[0], 4) * 100, dataframe[column].dtypes)
        columns_descriptions.append(column_description)

    data = pd.DataFrame(columns_descriptions, columns = ['FEATURE', 'USE', 'CARDINALITY', 'UNIQUE_DATA', 'VL_NULL', 'PC_NULL', 'TYPE'])

    data = data.sort_values(by='TYPE')
    data = data.reset_index(drop=True)

    return data

# drop some variables
def prod_variables_to_drop(dataframe, list_variables_to_drop):
    for column in list_variables_to_drop:
        try:
            dataframe = dataframe.drop(columns=column)
        except:
            pass

    return dataframe


# drop some variables
def prod_variables_to_drop_nan(dataframe, list_variables_to_drop_nan):
    for column in list_variables_to_drop_nan:
        try:
            dataframe = dataframe.drop(columns=column)
        except:
            pass

    return dataframe


# replace nan by mean
def prod_variables_to_fill_nan(dataframe, dict_to_fillna):
    for column, mean_value in dict_to_fillna.items():
        dataframe[column] = dataframe[column].fillna(mean_value)

    return dataframe


def prod_label_encoder(dataframe, dict_to_label_encoder):
    loaded_encoders = dict_to_label_encoder['encoders']
    loaded_columns = dict_to_label_encoder['columns']

    # Suponha test_df como sua base de teste
    for column in loaded_columns:
        if column in loaded_encoders:
            # Transforma a coluna usando o encoder carregado
            dataframe[column] = loaded_encoders[column].transform(dataframe[column])

    return dataframe


def prod_one_hot_encoder(dataframe, dict_to_one_hot_encoder):
    if len(dict_to_one_hot_encoder['columns']) > 0:
        loaded_encoder = dict_to_one_hot_encoder['encoder']
        loaded_columns = dict_to_one_hot_encoder['columns']

        # Suponha test_df como sua base de teste
        encoded_data = loaded_encoder.transform(dataframe[loaded_columns])
        encoded_cols = loaded_encoder.get_feature_names_out(loaded_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=dataframe.index)

        dataframe_aux = pd.concat([dataframe.drop(loaded_columns, axis=1), encoded_df], axis=1)

        return dataframe_aux
    else:
        return dataframe
    

def prod_correlation(dataframe, list_of_variables_to_keep_low_correlation):
    #
    dataframe = dataframe[list_of_variables_to_keep_low_correlation]
    
    return dataframe

# scale
def prod_scaler(dataframe, scaler):
    # Suponha dataframe como sua base de teste
    dataframe_scaled = scaler.transform(dataframe)
    dataframe = pd.DataFrame(dataframe_scaled, columns=dataframe.columns, index=dataframe.index)

    return dataframe

# feature selection
def prod_feature_selection(dataframe, list_of_selected_features):
    dataframe_aux = dataframe[list_of_selected_features]

    return dataframe_aux


def calcular_ks_statistic(y_true, y_score):
    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values(by='score', ascending=False)
    total_events = df.target.sum()
    total_non_events = len(df) - total_events
    df['cum_events'] = df.target.cumsum()
    df['cum_non_events'] = (df.target == 0).cumsum()
    df['cum_events_percent'] = df.cum_events / total_events
    df['cum_non_events_percent'] = df.cum_non_events / total_non_events
    ks_statistic = np.abs(df.cum_events_percent - df.cum_non_events_percent).max()
    return ks_statistic


def avaliar_modelo(X_train, y_train, X_test, y_test, modelo, nm_modelo, root_path, nome_pasta_pai):

    feature_names = list(X_train.columns)
    # Criação da figura e dos eixos
    fig, axs = plt.subplots(5, 2, figsize=(15, 30))  # Ajustado para incluir novos gráficos
    plt.tight_layout(pad=6.0)

    # Cor azul claro
    cor = 'skyblue'

    # Taxa de Evento e Não Evento
    event_rate = np.mean(y_train)
    non_event_rate = 1 - event_rate
    axs[0, 0].bar(['Evento', 'Não Evento'], [event_rate, non_event_rate], color=[cor, 'lightcoral'])
    axs[0, 0].set_title('Taxa de Evento e Não Evento')
    axs[0, 0].set_ylabel('Proporção')

    # Importância dos Atributos
    importancias = None
    if hasattr(modelo, 'coef_'):
        importancias = np.abs(modelo.coef_[0])
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_

    if importancias is not None:
        importancias_df = pd.DataFrame({'feature': feature_names, 'importance': importancias})
        importancias_df = importancias_df.sort_values(by='importance', ascending=True)

        axs[0, 1].barh(importancias_df['feature'], importancias_df['importance'], color=cor)
        axs[0, 1].set_title('Importância das Variáveis - ' + nm_modelo)
        axs[0, 1].set_xlabel('Importância')

    else:
        axs[0, 1].axis('off')  # Desativa o subplot se não houver importâncias para mostrar

    # Confusion Matrix - Treino
    y_pred_train = modelo.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    axs[1, 0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix - Treino - ' + nm_modelo)
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels(['0', '1'])
    axs[1, 0].set_yticklabels(['0', '1'])
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        axs[1, 0].text(j, i, format(cm_train[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    # Confusion Matrix - Teste
    y_pred_test = modelo.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    axs[1, 1].imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 1].set_title('Confusion Matrix - Teste - ' + nm_modelo)
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(['0', '1'])
    axs[1, 1].set_yticklabels(['0', '1'])
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        axs[1, 1].text(j, i, format(cm_test[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    # ROC Curve - Treino e Teste
    y_score_train = modelo.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    axs[2, 0].plot(fpr_train, tpr_train, color=cor, label='Treino')

    y_score_test = modelo.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    axs[2, 0].plot(fpr_test, tpr_test, color='darkorange', label='Teste')

    axs[2, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[2, 0].set_title('ROC Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 0].set_xlabel('False Positive Rate')
    axs[2, 0].set_ylabel('True Positive Rate')
    axs[2, 0].legend(loc="lower right")

    # Precision-Recall Curve - Treino e Teste
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    axs[2, 1].plot(recall_train, precision_train, color=cor, label='Treino')

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_score_test)
    axs[2, 1].plot(recall_test, precision_test, color='darkorange', label='Teste')

    axs[2, 1].set_title('Precision-Recall Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 1].set_xlabel('Recall')
    axs[2, 1].set_ylabel('Precision')
    axs[2, 1].legend(loc="upper right")

    # Gini - Treino e Teste
    auc_train = roc_auc_score(y_train, y_score_train)
    gini_train = 2 * auc_train - 1
    auc_test = roc_auc_score(y_test, y_score_test)
    gini_test = 2 * auc_test - 1
    axs[3, 0].bar(['Treino', 'Teste'], [gini_train, gini_test], color=[cor, 'darkorange'])
    axs[3, 0].set_title('Gini - ' + nm_modelo)
    axs[3, 0].set_ylim(0, 1)
    axs[3, 0].text('Treino', gini_train + 0.01, f'{gini_train:.4f}', ha='center', va='bottom')
    axs[3, 0].text('Teste', gini_test + 0.01, f'{gini_test:.4f}', ha='center', va='bottom')

    # KS - Treino e Teste
    ks_train = calcular_ks_statistic(y_train, y_score_train)
    ks_test = calcular_ks_statistic(y_test, y_score_test)
    axs[3, 1].bar(['Treino', 'Teste'], [ks_train, ks_test], color=[cor, 'darkorange'])
    axs[3, 1].set_title('KS - ' + nm_modelo)
    axs[3, 1].set_ylim(0, 1)
    axs[3, 1].text('Treino', ks_train + 0.01, f'{ks_train:.4f}', ha='center', va='bottom')
    axs[3, 1].text('Teste', ks_test + 0.01, f'{ks_test:.4f}', ha='center', va='bottom')


    # # Decile Analysis - Teste
    # scores = modelo.predict_proba(X_test)[:, 1]
    # noise = np.random.uniform(0, 0.0001, size=scores.shape)  # Adiciona um pequeno ruído
    # scores += noise
    # deciles = pd.qcut(scores, q=10, duplicates='drop')
    # decile_analysis = y_test.groupby(deciles).mean()
    # axs[4, 1].bar(range(1, len(decile_analysis) + 1), decile_analysis, color='darkorange')
    # axs[4, 1].set_title('Ordenação do Score - Teste - ' + nm_modelo)
    # axs[4, 1].set_xlabel('Faixas de Score')
    # axs[4, 1].set_ylabel('Taxa de Evento')

    # # Decile Analysis - Treino
    # scores_train = modelo.predict_proba(X_train)[:, 1]
    # noise = np.random.uniform(0, 0.0001, size=scores_train.shape)  # Adiciona um pequeno ruído
    # scores_train += noise
    # deciles_train = pd.qcut(scores_train, q=10, duplicates='drop')
    # decile_analysis_train = y_train.groupby(deciles_train).mean()
    # axs[4, 0].bar(range(1, len(decile_analysis_train) + 1), decile_analysis_train, color=cor)
    # axs[4, 0].set_title('Ordenação do Score - Treino - ' + nm_modelo)
    # axs[4, 0].set_xlabel('Faixas de Score')
    # axs[4, 0].set_ylabel('Taxa de Evento')

    # Decile Analysis - Teste
    scores = modelo.predict_proba(X_test)[:, 0]
    noise = np.random.uniform(0, 0.0001, size=scores.shape)  # Adiciona um pequeno ruído
    scores += noise
    deciles = pd.qcut(scores, q=10, duplicates='drop')
    decile_analysis = y_test.groupby(deciles).mean()

    # Decile Analysis - Treino
    scores_train = modelo.predict_proba(X_train)[:, 0]
    noise = np.random.uniform(0, 0.0001, size=scores_train.shape)  # Adiciona um pequeno ruído
    scores_train += noise
    deciles_train = pd.qcut(scores_train, q=10, duplicates='drop')
    decile_analysis_train = y_train.groupby(deciles_train).mean()

    bar_width = 0.35  # Adjust the bar width as needed

    # Plot the bars for the test set
    axs[4, 0].bar(
        np.arange(1, len(decile_analysis) + 1),
        decile_analysis,
        width=bar_width,
        color='#654ffe',
        label='Teste'
    )

    # Plot the bars for the training set
    axs[4, 0].bar(
        np.arange(1, len(decile_analysis_train) + 1) + bar_width,
        decile_analysis_train,
        width=bar_width,
        color='#df4ffe',
        label='Treino'
    )

    # Set x-axis ticks to display counts from 0 to 10
    axs[4, 0].set_xticks(np.arange(1, 11))
    axs[4, 0].set_xticklabels(np.arange(1, 11))

    axs[4, 0].set_title('Ordenação do Score - ' + nm_modelo)
    axs[4, 0].set_xlabel('Faixas de Score')
    axs[4, 0].set_ylabel('Taxa de Evento')
    axs[4, 0].legend()

    # Salvar a imagem em um arquivo (por exemplo, formato PNG)
    plt.savefig(f'{root_path}/{nome_pasta_pai}/DATAS/GOLD/image_results_{nome_pasta_pai}.png')

    # Mostrar os gráficos
    plt.show()
    

def model_application(model, model_name, param_grid, X_train, y_train, X_test, cv=5, scoring=None):
    import time

    # Definindo os parâmetros para o grid search
    param_grid = param_grid

    # Calculando a quantidade total de modelos que serão treinados
    num_models = 1
    for key in param_grid:
        num_models = num_models * len(param_grid[key])

    # 5 é o número de folds na validação cruzada (cv)
    num_models = num_models * 5

    print(f"Total de Modelos a serem Treinados: {num_models}")

    # Registre o tempo inicial
    start_time = time.time()

    # Definindo o objeto GridSearchCV
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,  # 'accuracy' 'precision' 'recall' 'f1'
        n_jobs=-1
    )

    # Registre o tempo final
    end_time = time.time()

    # Calcule a diferença de tempo
    total_time = end_time - start_time

    # Treinando o modelo com o grid search
    grid.fit(X_train, y_train)

    # melhores parametros
    best_params = grid.best_params_

    # melhor score
    best_score = grid.best_score_

    # Exibindo os melhores parâmetros encontrados pelo grid search
    print("Melhores Parâmetros: ", best_params)

    # Exibindo a melhor pontuação (score) atingida pelo modelo com os melhores parâmetros
    print(f"Melhor {scoring}: ", best_score)

    # Utilizando o melhor modelo para fazer previsões
    predictions = grid.best_estimator_.predict(X_test)

    #
    best_model = grid.best_estimator_

    return model_name, num_models, best_model, best_params, best_score, predictions, total_time


def keep_low_correlation_variables(dataframe, correlation_threshold=0.8):
    # Calcula a matriz de correlação
    correlation_matrix = dataframe.corr().abs()

    # Obtém a lista de pares de variáveis altamente correlacionadas
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = (upper_triangle > correlation_threshold).any()

    # Lista de variáveis a serem mantidas (não removidas)
    variables_to_keep = [var for var in dataframe.columns if var not in highly_correlated_pairs.index or not highly_correlated_pairs[var]]
    
    return variables_to_keep


def objective(trial, X_train, X_test, y_train, y_test):
   
    params = {
        'tol': trial.suggest_uniform('tol', 1e-6, 1e-3),
        'C': trial.suggest_loguniform('C', 1e-2, 1),
        'penalty': trial.suggest_categorical('penalty', [None, 'l2']), # , 'elasticnet', None
        'solver': trial.suggest_categorical('solver', ['lbfgs']), # , 'saga', 'liblinear'
        'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'n_jobs': -1
    }

    model = LogisticRegression(**params, random_state=42)
    
    # # Ajuste o modelo usando validação cruzada e retorne a média da pontuação
    # score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    
    model.fit(X_train, y_train)
    y_predict = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_predict)

    return roc_auc