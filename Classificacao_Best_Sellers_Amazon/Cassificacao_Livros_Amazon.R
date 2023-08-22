# Definindo o diretório de trabalho
setwd("D:/Data Science/Projetos/R com Azure/Projetos-em-linguagem-R-com-Azure/Classificacao_Best_Sellers_Amazon")
getwd()

# Carregando as bibliotecas
library(dplyr)
library(ggplot2)
library(readxl)
library(Amelia)
library(caTools)
library(class)
library(randomForest)
library(e1071)
library(pROC)
library(nnet)


# Carregando o dataset, disponível em: https://www.kaggle.com/datasets/abdulhamidadavize/top-100-best-selling-books-on-amazon-20092021?resource=download
dados <- read_excel('Amazon_top100_bestselling_books_2009to2021.xlsx')

# Dimensões
dim(dados)

#visualizando os dados
View(dados)

# Variáveis e Tipos de Dados
str(dados)

# Sumários das variáveis numéricas
summary(dados)

################## Analise exploratória de dados ####################

# Retirando a primeira coluna com a posição do livro (ordem de raspagem)
df <- dados[, -1]
str(df)

# Nomes das colunas
colnames(df)

# Criando o cabeçalho
cabecalho <- c('Preco', 'Rank', 'Titulo', 'Review', 'Avaliacao', 'Autor', 'Capa', 'Ano', 'Genero')
cabecalho

# Renomeando o cabeçalho no dataframe
colnames(df) <- cabecalho
colnames(df)
rm(cabecalho)

# Verificando NA
missmap(df, 
        main = "Best-sellers - Mapa de Dados Missing", 
        col = c("yellow", "black"), 
        legend = FALSE)

# Quantas linhas tem casos completos?
complete_cases <- sum(complete.cases(df))

# Quantas linhas tem casos incompletos?
not_complete_cases <- sum(!complete.cases(df))

# Qual o percentual de dados incompletos?
percentual <- (not_complete_cases / complete_cases) * 100
percentual
rm(complete_cases)
rm(not_complete_cases)
rm(percentual)

# Verificando os livros que foram excluídos devido a falta de informação
# Verificando quantos livros únicos há no dataset
titulo <- unique(df$Titulo)
df <- na.omit(df)
titulo_sem_na <- unique(df$Titulo)
setdiff(titulo, titulo_sem_na)
rm(titulo)
rm(titulo_sem_na)

# Verificando livros duplicados
livros_repetidos <- subset(df$Titulo, duplicated(df) | duplicated(df$Titulo, fromLast = TRUE))
livros_repetidos

# Excluindo os livros repetidos e ficando com o mais recente
df_limpos <- df %>%
  distinct(Titulo, .keep_all = TRUE)

# Retirando os livros com generos desconhecidos
df_limpos <- df_limpos %>%
  filter(as.character(Genero) != 'unknown')

# Transformando a variável genero para binário (1 para Fiction e 0 para No Fiction)
df_limpos$Genero <- ifelse(df_limpos$Genero == 'Fiction', 1, 0)

# Transformando as strings para variável categóricas
# Variáveis para conversão
Var_cat <- c('Autor', 'Capa', 'Ano', 'Rank', 'Avaliacao')

# Convertendo as variáveis 
df_limpos[, Var_cat] <- lapply(df_limpos[, Var_cat], factor)
rm(Var_cat)
str(df_limpos)

# Vizualizando os dados
ggplot(df_limpos,aes(Avaliacao)) + geom_bar(aes(fill = factor(Avaliacao)), alpha = 0.5)
ggplot(df_limpos,aes(Autor)) + geom_bar(aes(fill = factor(Autor)), alpha = 0.5)
ggplot(df_limpos, aes(Capa)) + geom_bar(aes(fill = factor(Capa)), alpha = 0.5)
ggplot(df_limpos, aes(Ano)) + geom_bar(aes(fill = factor(Ano)), alpha = 0.5)
ggplot(df_limpos,aes(Genero)) + geom_histogram(fill = 'green', color = 'black', alpha = 0.5)

# Criando um função de normalização
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalizando
colunas_para_normalizar <- c('Preco','Review')
df_norm <- df_limpos %>%
  mutate(across(all_of(colunas_para_normalizar), normalizar))
rm(colunas_para_normalizar)
str(df_norm)

##################################Machine Learning ######################################

#Separando o dataset em treino e teste
# Defina a proporção de divisão entre treino e teste
proporcao_treino <- 0.75

# Calcule o número de amostras para treino e teste
num_amostras_treino <- round(nrow(df_limpos) * proporcao_treino)
num_amostras_teste <- nrow(df_limpos) - num_amostras_treino

# Amostra índices aleatórios para treino e teste
indices_treino <- sample(seq_len(nrow(df_limpos)), size = num_amostras_treino)
indices_teste <- setdiff(seq_len(nrow(df_limpos)), indices_treino)

# Divida o dataframe em treino e teste
df_treino <- df_limpos[indices_treino, ]
df_teste <- df_limpos[indices_teste, ]
rm(num_amostras_teste)
rm(num_amostras_treino)
rm(proporcao_treino)
str(df_treino)

# Criando o modelo de Bayes
nb_model <- naiveBayes(Genero ~ . - Rank - Titulo, data = df_treino)
nb_model_v1 <- naiveBayes(Genero ~., data=df_treino)
nb_model_v2 <- naiveBayes(Genero ~ . - Titulo - Rank - Capa , data=df_treino)

# Visualizando o resultado
nb_model
summary(nb_model)
str(nb_model)

# Faça as Previsões
nb_test_predict <- predict(nb_model, df_teste)
nb_test_predict_v1 <- predict(nb_model_v1, df_teste)
nb_test_predict_v2 <- predict(nb_model_v2, df_teste)

# Crie a Confusion matrix
table(pred = nb_test_predict, true = df_teste$Genero)
table(pred = nb_test_predict_v1, true = df_teste$Genero)
table(pred = nb_test_predict_v2, true = df_teste$Genero)

# Média
mean(nb_test_predict == df_teste$Genero)
mean(nb_test_predict_v1 == df_teste$Genero)
mean(nb_test_predict_v2 == df_teste$Genero)

# Gerando a curva ROC
curva_roc <- roc(df_teste$Genero, as.numeric(nb_test_predict))
plot(curva_roc, main = "Curva ROC - Naive Bayes",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_v1 <- roc(df_teste$Genero, as.numeric(nb_test_predict_v1))
plot(curva_roc_v1, main = "Curva ROC v1 - Naive Bayes",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_v2 <- roc(df_teste$Genero, as.numeric(nb_test_predict_v2))
plot(curva_roc_v2, main = "Curva ROC v2 - Naive Bayes",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc <- auc(curva_roc)
cat("AUC-ROC:", auc_roc, "\n") #0.8278

auc_roc_v1 <- auc(curva_roc_v1)
cat("AUC-ROC:", auc_roc_v1, "\n") #0.7549

auc_roc_v2 <- auc(curva_roc_v2)
cat("AUC-ROC:", auc_roc_v2, "\n") #0.8329

# Criando o modelo por Random Forest
modelo_rf <- randomForest(factor(Genero) ~ . - Rank - Titulo - Autor, data = df_treino, ntree = 100)

# Faça as Previsões RF
nb_test_predict_rf <- predict(modelo_rf, df_teste)


# Crie a Confusion matrix
table(pred = nb_test_predict_rf, true = df_teste$Genero)


# Média
mean(nb_test_predict_rf == df_teste$Genero)


# Gerando a curva ROC
curva_roc_rf <- roc(df_teste$Genero, as.numeric(nb_test_predict_rf))
plot(curva_roc_rf, main = "Curva ROC - Random Forest",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_rf <- auc(curva_roc_rf)
cat("AUC-ROC:", auc_roc_rf, "\n") #0.7189

# Criando o modelo logístico
modelo <- multinom(Genero ~ Autor + Preco + Ano + Avaliacao + Rank + Capa + Review, family = "binomial", data = df_treino)
modelo_v1 <- multinom(Genero ~ Autor + Preco + Ano, family = "binomial", data = df_treino)

# Faça as Previsões RF
nb_test_predict_log <- predict(modelo, df_teste)
nb_test_predict_log_v1 <- predict(modelo_v1, df_teste)

# Crie a Confusion matrix
table(pred = nb_test_predict_log, true = df_teste$Genero)
table(pred = nb_test_predict_log_v1, true = df_teste$Genero)

# Média
mean(nb_test_predict_log == df_teste$Genero)
mean(nb_test_predict_log_v1 == df_teste$Genero)

# Gerando a curva ROC
curva_roc_log <- roc(df_teste$Genero, as.numeric(nb_test_predict_log))
plot(curva_roc_log, main = "Curva ROC - Logistica",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

curva_roc_log_v1 <- roc(df_teste$Genero, as.numeric(nb_test_predict_log_v1))
plot(curva_roc_log_v1, main = "Curva ROC - Logistica v1",
     xlab = "Taxa de Falsos Positivos",
     ylab = "Taxa de Verdadeiros Positivos")

# Calcular e imprimir a AUC-ROC
auc_roc_log <- auc(curva_roc_log)
cat("AUC-ROC:", auc_roc_log, "\n") #0.7549

auc_roc_log_v1 <- auc(curva_roc_log_v1)
cat("AUC-ROC:", auc_roc_log_v1, "\n") #0.7544

