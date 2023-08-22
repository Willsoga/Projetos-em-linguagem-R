# Definindo o diretório de trabalho
setwd("D:/Data Science/Projetos/R com Azure/Projetos-em-linguagem-R-com-Azure/Preisao_eficiencia_extintor")
getwd()

# Carregando as bibliotecas
library(dplyr)
library(ggplot2)
library(readxl)
library(Amelia)
library(skimr)
library(tidyr)
library(corrplot)
library(gmodels)
library(hrbrthemes)
library(viridis)
library(caTools)
library(pROC)
library(e1071)
library(randomForest)

# Carregando os dados
dados <- read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')

# Dimensões
dim(dados)

#visualizando os dados
View(dados)

# Variáveis e Tipos de Dados
str(dados)

# Sumários das variáveis numéricas
summary(dados)

################## Analise exploratória de dados ####################
# Nomes das colunas
colnames(dados)

#Criando um cabeçalho
cabecalho <- c('tamanho', 'combustivel', 'distancia', 'decibel', 'fluxo_de_ar', 'frequencia', 'status')
cabecalho

# Renomeando o cabeçalho no dataframe
colnames(dados) <- cabecalho
colnames(dados)
rm(cabecalho)
str(dados)

# Verificando os dados
head(dados)
tail(dados)

# Verificando NA
missmap(dados, 
        main = "Teste de eficiencia de extintor - Mapa de Dados Missing", 
        col = c("yellow", "black"), 
        legend = FALSE)

# Verificando a variável categórica
dados %>% count(combustivel)
ggplot(dados,aes(combustivel)) + geom_bar(aes(fill = combustivel), alpha = 0.5) +
  geom_text(aes(label = after_stat(count), y = ..count..), 
            stat = "count",  
            position = position_dodge(width = 0.9),  
            vjust = -0.5)

#Alterando o nome do combustível
dados$combustivel[dados$combustivel == 'gasoline'] <- 'gasolina'
dados$combustivel[dados$combustivel == 'kerosene'] <- 'querosene'
dados$combustivel[dados$combustivel == 'thinner'] <- 'tiner'

# Verificando o relacionamento das duas variáveis categóricas
CrossTable(x = dados$combustivel, y = dados$status, chisq = TRUE)
chisq.test(x = dados$combustivel, y = dados$status) #p-value < 0.05, rejeitamos a H0, ou seja, há associação entre os grupos

# Plot
ggplot(data=dados, aes(x=combustivel, group = status, fill = status)) +
  geom_bar(position ='dodge', alpha=.4) +
  geom_text(
  aes(label = after_stat(count), y = ..count..), 
  stat = "count",  
  position = position_dodge(width = 0.9),  
  vjust = -0.5)

# Alterando a coluna combustível para binario
df2 <- model.matrix(~ combustivel - 1, dados)
colnames(df) <- gsub("combustivel", "", colnames(df2))
df <- cbind(dados, df2)
rm(df2)
df <- df[,-2]
View(df)
df %>% count(gasolina)
df %>% count(querosene)
df %>% count(lpg)
df %>% count(tiner)

# Sumário
df |> skim()

# Verificando a correlação
options(digits = 3)
correlacao <- cor(df)
corrplot(correlacao, method = 'color')

# Definindo intervalos das variáveis númericas

# Classificação de acordo com os quartis
tamanho.cut <- cut(df$tamanho, breaks =  quantile(df$tamanho),
                   include.lowest = TRUE)


distancia.cut <- cut(df$distancia, breaks = quantile(df$distancia),
                     include.lowest = TRUE)

decibel.cut <- cut(df$decibel, breaks = quantile(df$decibel),
                     include.lowest = TRUE)

fluxo.cut <- cut(df$fluxo_de_ar, breaks = quantile(df$fluxo_de_ar),
                 include.lowest = TRUE)

frequencia.cut <- cut(df$frequencia, breaks = quantile(df$frequencia),
                      include.lowest = TRUE)

# Tabela de frequências absolutas
status_tamanho <- data.frame(table(df$status, tamanho.cut))
status_distancia <- data.frame(table(df$status, distancia.cut))
status_decibel <- data.frame(table(df$status, decibel.cut))
status_fluxo <- data.frame(table(df$status, fluxo.cut))
status_frequencia <- data.frame(table(df$status, frequencia.cut))

#Plot
ggplot(data = status_decibel, aes(x = decibel.cut, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(x = "Intervalo", y = "Frequência", title = "Histograma de Decibel")

ggplot(data = status_tamanho, aes(x = tamanho.cut, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(x = "Intervalo", y = "Frequência", title = "Histograma de Tamanhos")

ggplot(data = status_distancia, aes(x = distancia.cut, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(x = "Intervalo", y = "Frequência", title = "Histograma de Distancia")

ggplot(data = status_fluxo, aes(x = fluxo.cut, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(x = "Intervalo", y = "Frequência", title = "Histograma de Fluxo")

ggplot(data = status_frequencia, aes(x = frequencia.cut, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(x = "Intervalo", y = "Frequência", title = "Histograma de Frequencia")

############################### Amostragem ###################################

# Definir a semente para reproduzibilidade
set.seed(104)

dim(df)

# Tamanho da amostra
tamanho_amostra <- 1000

# Criar a amostra aleatória
indice <- sample(1:nrow(df), size = tamanho_amostra, replace = FALSE)
amostra <- df[indice, ]
View(amostra)

# Definindo os novos dados para teste
dados_teste_final <- df[-indice, ]

################################ Machine Learning ###########################

# Treine o modelo e depois faça as previsões
log.model <- glm(formula = status ~ . , family = binomial(link = 'logit'), data = amostra)
log.model_v1 <- glm(formula = status ~ . -combustivelgasolina - combustiveltiner , family = binomial(link = 'logit'), data = amostra)
log.model_v2 <- glm(formula = status ~ . -combustivelgasolina - combustiveltiner - decibel, family = binomial(link = 'logit'), data = amostra)

# Verificando a importância das variáveis
summary(log.model)
summary(log.model_v1)
summary(log.model_v2)

# Split dos dados
split = sample.split(amostra$status, SplitRatio = 0.75)

# Datasets de treino e de teste
dados_treino = subset(amostra, split == TRUE)
dados_teste = subset(amostra, split == FALSE)

# Gerando o modelo com o dataset de treino
log.model_v3 <- glm(formula = status ~ . -combustivelgasolina - combustiveltiner - decibel, family = binomial(link = 'logit'), data = dados_treino)
log.model_v4 <- glm(formula = status ~ . -combustivelgasolina - combustiveltiner, family = binomial(link = 'logit'), data = dados_treino)

# Verificando a importância das variáveis
summary(log.model_v3)
summary(log.model_v4)

# Prevendo a acurácia
fitted.probabilities <- predict(log.model_v3, newdata = dados_teste, type = 'response')
fitted.probabilities_v4 <- predict(log.model_v4, newdata = dados_teste, type = 'response')

# Calculando os valores
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)
fitted.results_v4 <- ifelse(fitted.probabilities_v4 > 0.5, 1, 0)

# Acurácia 
misClasificError <- mean(fitted.results != dados_teste$status)
print(paste('Acuracia', 1-misClasificError)) #0,912
misClasificError_v4 <- mean(fitted.results_v4 != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_v4)) #0.908


# Criando a confusion matrix
table(dados_teste$status, fitted.probabilities > 0.5)
table(dados_teste$status, fitted.probabilities_v4 > 0.5)

# Gerando a curva ROC
curva_roc_log <- roc(dados_teste$status, fitted.probabilities_v4)
plot(curva_roc_log, main = "Curva ROC - Logistica",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_log_v4 <- roc(dados_teste$status, fitted.probabilities_v4)
plot(curva_roc_log_v4, main = "Curva ROC - Logistica_v4",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_log <- auc(curva_roc_log)
cat("AUC-ROC:", auc_roc_log, "\n") #0.965

auc_roc_log_v4 <- auc(curva_roc_log_v4)
cat("AUC-ROC:", auc_roc_log_v4, "\n") #0.967

# Criando o modelo de Bayes
nb_model <- naiveBayes(status ~ . -combustivelgasolina - combustiveltiner , data = dados_treino)
nb_model_v1 <- naiveBayes(status ~., data=dados_treino)
nb_model_v2 <- naiveBayes(status ~ . -combustivelgasolina - combustiveltiner - decibel , data=dados_treino)
nb_model_v3 <- naiveBayes(status ~ distancia + fluxo_de_ar, data = dados_treino)

# Prvisões
nb_test_predict <- predict(nb_model, dados_teste)
nb_test_predict_v1 <- predict(nb_model_v1, dados_teste)
nb_test_predict_v2 <- predict(nb_model_v2, dados_teste)
nb_test_predict_v3 <- predict(nb_model_v3, dados_teste)

# Crie a Confusion matrix
table(pred = nb_test_predict, true = dados_teste$status)
table(pred = nb_test_predict_v1, true = dados_teste$status)
table(pred = nb_test_predict_v2, true = dados_teste$status)
table(pred = nb_test_predict_v3, true = dados_teste$status)

# Acurácia
misClasificError_nb <- mean(nb_test_predict != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_nb)) #0,876
misClasificError_nb_v1 <- mean(nb_test_predict_v1 != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_nb_v1)) #0.884
misClasificError_nb_v2 <- mean(nb_test_predict_v2 != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_nb_v2)) #0.876
misClasificError_nb_v3 <- mean(nb_test_predict_v3 != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_nb_v3)) #0.86

# Gerando a curva ROC
curva_roc_nb <- roc(dados_teste$status, as.numeric(nb_test_predict))
plot(curva_roc_nb, main = "Curva ROC - Naive Bayes",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_nb_v1 <- roc(dados_teste$status, as.numeric(nb_test_predict_v1))
plot(curva_roc_nb_v1, main = "Curva ROC - Naive Bayes_v1",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_nb_v2 <- roc(dados_teste$status, as.numeric(nb_test_predict_v2))
plot(curva_roc_nb_v2, main = "Curva ROC - Naive Bayes_v2",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_nb_v3 <- roc(dados_teste$status, as.numeric(nb_test_predict_v3))
plot(curva_roc_nb_v3, main = "Curva ROC - Naive Bayes_v3",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_nb <- auc(curva_roc_nb)
cat("AUC-ROC:", auc_roc_nb, "\n") #0.876

auc_roc_nb_v1 <- auc(curva_roc_nb_v1)
cat("AUC-ROC:", auc_roc_nb_v1, "\n") #0.884

auc_roc_nb_v2 <- auc(curva_roc_nb_v2)
cat("AUC-ROC:", auc_roc_nb_v2, "\n") #0.876

auc_roc_nb_v3 <- auc(curva_roc_nb_v3)
cat("AUC-ROC:", auc_roc_nb_v3, "\n") #0.86

# Criando o modelo por Random Forest
modelo_rf <- randomForest(status ~ ., data = dados_treino, ntree = 100)
modelo_rf_v1 <- randomForest(status ~ . -combustivelgasolina - combustiveltiner, data = dados_treino, ntree = 100)
modelo_rf_v2 <- randomForest(status ~ ., data = dados_treino, ntree = 90)

# Faça as Previsões RF
nb_test_predict_rf <- predict(modelo_rf, dados_teste)
nb_test_predict_rf_v1 <- predict(modelo_rf_v1, dados_teste)
nb_test_predict_rf_v2 <- predict(modelo_rf_v2, dados_teste)

# Crie a Confusion matrix
table(pred = round(nb_test_predict_rf), true = dados_teste$status)
table(pred = round(nb_test_predict_rf_v1), true = dados_teste$status)
table(pred = round(nb_test_predict_rf_v2), true = dados_teste$status)

# Acurácia
misClasificError_rf <- mean(round(nb_test_predict_rf) != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_rf)) #0,928
misClasificError_rf_v1 <- mean(round(nb_test_predict_rf_v1) != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_rf_v1)) #0.92
misClasificError_rf_v2 <- mean(round(nb_test_predict_rf_v2) != dados_teste$status)
print(paste('Acuracia', 1-misClasificError_rf_v2)) #0.928

# Gerando a curva ROC
curva_roc_rf <- roc(dados_teste$status, round(nb_test_predict_rf))
plot(curva_roc_rf, main = "Curva ROC - Random Forest",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_rf_v1 <- roc(dados_teste$status, round(nb_test_predict_rf_v1))
plot(curva_roc_nb_v1, main = "Curva ROC - Random Forest_v1",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_rf_v2 <- roc(dados_teste$status, round(nb_test_predict_rf_v2))
plot(curva_roc_rf, main = "Curva ROC - Random Forest_v2",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_rf <- auc(curva_roc_rf)
cat("AUC-ROC:", auc_roc_rf, "\n") #0.928

auc_roc_rf_v1 <- auc(curva_roc_rf_v1)
cat("AUC-ROC:", auc_roc_rf_v1, "\n") #0.92

auc_roc_rf_v2 <- auc(curva_roc_rf_v2)
cat("AUC-ROC:", auc_roc_rf_v2, "\n") #0.928

############################# Aplicando o modelo nos dados não visto #########################

# Previsões
nb_test_predict_rf_vf <- predict(modelo_rf, dados_teste_final)
fitted.probabilities_vf <- predict(log.model_v4, newdata = dados_teste_final, type = 'response')
fitted.results_vf <- ifelse(fitted.probabilities_vf > 0.5, 1, 0)

# Crie a Confusion matrix
table(pred = round(nb_test_predict_rf_vf), true = dados_teste_final$status)
table(dados_teste_final$status, fitted.probabilities_vf > 0.5)

# Acurácia
misClasificError_rf_vf <- mean(round(nb_test_predict_rf_vf) != dados_teste_final$status)
print(paste('Acuracia', 1-misClasificError_rf_vf)) #0.922
misClasificError_vf <- mean(fitted.results_vf != dados_teste_final$status)
print(paste('Acuracia', 1-misClasificError_vf)) #0.897

# ROC
curva_roc_rf_vf <- roc(dados_teste_final$status, round(nb_test_predict_rf_vf))
plot(curva_roc_rf_vf, main = "Curva ROC - Random Forest",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_log_vf <- roc(dados_teste_final$status, fitted.probabilities_vf)
plot(curva_roc_log_vf, main = "Curva ROC - Logistica",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_rf_vf <- auc(curva_roc_rf_vf)
cat("AUC-ROC:", auc_roc_rf_vf, "\n") #0.922

auc_roc_log_vf <- auc(curva_roc_log_vf)
cat("AUC-ROC:", auc_roc_log_vf, "\n") #0.965
