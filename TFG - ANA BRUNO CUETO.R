#ANALISIS DEL PRECIO DE LA VIVIENDA EN LOS DISTRITOS DE CIUDAD LINEAL Y LA LATINA CON MODELOS DE MACHINE LEARNING

#ANA BRUNO CUETO

#LIBRERIAS
library(readr)
library(ggplot2)
library(Hmisc)
library(corrplot)
library(plyr)
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(MASS)
library(stats)
library(RWeka)
library(neuralnet)
library(skimr)
library(DataExplorer)
library(ggpubr)
library(mosaicData)
library(h2o)
library(dplyr)
library(recipes)
library(NeuralNetTools)
library(nnet)
library(kernlab)
library(ranger)
library(caretEnsemble)
library(tictoc)

#eliminamos espacios en blanco de variable planta
PISOS$Planta <- str_trim(PISOS$Planta, side = "both")
table(PISOS$Planta)

## Convertir en factor las variables categóricas y en numeros enteros las variables Banos y Habitaciones

PISOS$Planta<-factor(PISOS$Planta, levels=c("-1","-2", "1", "10","11","12","13","14","15","16","2","3","4","5","6","7","8","9","Bajo","Desconocido","Entreplanta","Semi-sótano","Sotano"))

PISOS$Distrito<-factor(PISOS$Distrito, levels=c("Ciudad Lineal", "Latina"))

PISOS$Garaje<-factor(PISOS$Garaje, levels=c("Garaje opc. 10.000€","Garaje opc. 12.000€","Garaje opc. 15.000 €","Garaje opc. 18.000 €","Garaje opc. 19.000 €","Garaje opc. 20.000 €","Garaje opc. 24.000 €","Garaje opc. 25.000 €","Garaje opc. 26.000 €", "Garaje opc. 30.000 €","Garaje opc. 50.000 €","Garaje opc. 8.518 €","No","Si"))

PISOS$Ascensor<-factor(PISOS$Ascensor, levels=c("No", "Si"))

PISOS$Exterior.Interior<-factor(PISOS$Exterior.Interior, levels=c("Desconocido", "Exterior","Interior"))

PISOS$Terraza.Balcon<-factor(PISOS$Terraza.Balcon, levels=c("Balcon", "Ninguno", "Terraza", "Terraza y Balcon"))

PISOS$Calefaccion<-factor(PISOS$Calefaccion, levels=c("Calefaccion central", "Calefaccion central: Gas", "Calefaccion central: Gas natural", "Calefaccion central: Gasoil", "Calefaccion individual", "Calefaccion individual: Bomba de frío/calor", "Calefaccion individual: Eléctrica", "Calefaccion individual: Gas natural", "Calefaccion individual: Gas propano/butano", "Desconocido", "No dispone de Calefaccion"))

PISOS$Estado<-factor(PISOS$Estado, levels=c("Desconocido", "Promoción de obra nueva", "Segunda mano/buen estado", "Segunda mano/para reformar"))

PISOS$Habitaciones<-as.integer(PISOS$Habitaciones)
PISOS$Banos<-as.integer(PISOS$Banos)

#1.ANALISIS EXPLORATORIO DE DATOS

#1.1.Categorización de las variables y tratamiento de valores pérdidos

str(PISOS)

#Valores pérdidos

sum(is.na(PISOS))

colSums(is.na(PISOS))
colMeans(is.na(PISOS), round(2))

# Calcular la media de la columna con valores perdidos
col_mean_banos <- mean(PISOS$Banos, na.rm = TRUE)
mean_banos_rounded <- round(col_mean_banos, 0)

col_mean_habitaciones <- mean(PISOS$Habitaciones, na.rm = TRUE)
mean_habitaciones_rounded <- round(col_mean_habitaciones, 0)

# Reemplazar los valores perdidos por la media
PISOS$Banos[is.na(PISOS$Banos)] <- mean_banos_rounded 
PISOS$Habitaciones[is.na(PISOS$Habitaciones)] <- mean_habitaciones_rounded

PISOS$Habitaciones<-as.integer(PISOS$Habitaciones)
PISOS$Banos<-as.integer(PISOS$Banos)
str(PISOS)

# Reemplazar los valores perdidos de Garaje por "No hay garaje"
PISOS$Garaje[is.na(PISOS$Garaje)] <- "No"

# Reemplazar los valores perdidos de Terraza/Balcón por "No hay terraza ni balcón"
PISOS$Terraza.Balcon[is.na(PISOS$Terraza.Balcon)] <- "Ninguno"

# Reemplazar los valores perdidos de Estado por "Desconocido"
PISOS$Estado[is.na(PISOS$Estado)] <- "Desconocido"

# Reemplazar los valores perdidos de Exterior/Interior por "Desconocido"
PISOS$Exterior.Interior[is.na(PISOS$Exterior.Interior)] <- "Desconocido"

# Reemplazar los valores perdidos de Planta por "Desconocido"
PISOS$Planta[is.na(PISOS$Planta)] <- "Desconocido"

# Reemplazar los valores perdidos de Ascensor por "No"
PISOS$Ascensor[is.na(PISOS$Ascensor)] <- "No"

# Reemplazar los valores perdidos de Calefaccion por "Desconocido"
PISOS$Calefaccion[is.na(PISOS$Calefaccion)] <- "Desconocido"

#Comprobar que ya no tienen valores perdidos las variables mencionadas.
colSums(is.na(PISOS))

sum(is.na(PISOS)) #ahora es 0

plot_missing(
  data    = PISOS, 
  title   = "Porcentaje de valores ausentes",
  ggtheme = theme_bw(),
  theme_config = list(legend.position = "none")
)

#1.2 Analisis descriptivo

summary(PISOS)

#Variables numéricas

#Histograma Precio

options(scipen = 999)

precio<-(PISOS$Precio)

hist(precio, main = "Histograma de Precios de Viviendas", breaks = 80,
     xlab = "Precio de Viviendas (en EUR)",ylab="Frecuencia", col = "blue")

#Histograma M2

M2<-(PISOS$M2)

hist(M2, main = "Histograma de M2 de Viviendas", breaks = 80,
     xlab = "Metros cuadrados", ylab="Frecuencia", col = "red")

#Histograma Habitaciones

habitaciones<-(PISOS$Habitaciones)

hist(habitaciones, main = "Histograma de habitaciones de Viviendas", breaks = 7,
     xlab = "Nº de habitaciones", ylab="Frecuencia", col = "yellow")

#Histograma Banos

banos<-(PISOS$Banos)

hist(banos, main = "Histograma de banos de Viviendas", breaks = 8,
     xlab = "Nº de banos", ylab="Frecuencia", col = "green")

#1.4.	Identificación y eliminacion de valores atípicos 

iqr_precios <- IQR(PISOS$Precio)
limite_superior <- quantile(PISOS$Precio, 0.75) + 1.5 * iqr_precios
limite_inferior <- quantile(PISOS$Precio, 0.25) - 1.5 * iqr_precios

PISOS_filtrado <- filter(PISOS, PISOS$Precio >= limite_inferior & PISOS$Precio <= limite_superior)

boxplot(PISOS$Precio)
boxplot(PISOS_filtrado$Precio)

str(PISOS_filtrado)

#1.5.	Correlacion entre las variables

variables_numericas <- PISOS_filtrado[,c(2,3,4,8)]

correlacion <- cor(variables_numericas)

correlacion

corrplot(correlacion, method = "pie")

#2. MODELOS DE MACHINE LEARNING

precio_medio <- mean(PISOS_filtrado$Precio)
precio_medio

RMSE_media<-RMSE(precio_medio,PISOS_filtrado$Precio)
RMSE_media

#2.1. REGRESION LINEAL MULTIPLE EXPLICATIVA

str(PISOS_filtrado)

modelo1.explicativo <- lm(Precio ~ ., data = PISOS_filtrado)
modelo1.explicativo

summary(modelo1.explicativo)

#2.2. REGRESION MULTIPLE PREDICTIVA

Pisos2<-data.frame(PISOS_filtrado[,c(1,2,3,4,5,6,8,10,11)])
str(Pisos2)

RNGkind("Super", "Inversion") 
set.seed(123) 
n<-nrow(Pisos2)

#PARTICIÓN 80% TRAIN, 30% TEST
TrainIndex<-sample(1:n, n*0.8) 
TrainingSet<-Pisos2[TrainIndex,] 
TestSet<-Pisos2[-TrainIndex,]

# estimamos modelo de regresion sobre el TrainingSet con todas las variables seleccionadas
modelo1_predictivo<- lm(Precio~ ., data=TrainingSet)

summary(modelo1_predictivo)

# obtenemos predicciones en el TestSet
predicciones<-predict(modelo1_predictivo, newdata = TestSet)

# calculamos los errores de predicción 
errores<-TestSet$Precio-predicciones
h<-nrow(TestSet) 

# calculo de las medidas de performance predictiva en el TestSet 
RMSE<-sqrt(sum(errores^2)/h)
RMSE

#representacion grafica de los errores
datos_grafica_predictivo1 <- data.frame(TestSet$Precio, predicciones)

ggplot(datos_grafica_predictivo1, aes(x = TestSet$Precio, y = predicciones)) + 
  geom_point(color = "red") +  
  geom_abline(intercept = 0, slope = 1, color = "black")+
  labs(title = "Relación entre el precio real y el predicho", x = "Precio real", y = "Predicciones")

desviacion_estandar <- sd(Pisos2$Precio)
desviacion_estandar

#modificaciones para mejorar el modelo
Pisos3 <- Pisos2
str(Pisos3)

Pisos3$M2_2 <- Pisos3$M2^2
Pisos3$Banos2 <- Pisos3$Banos^2

set.seed(123) 
n<-nrow(Pisos3)

#PARTICIÓN 80% TRAIN, 30% TEST
TrainIndex3<-sample(1:n, n*0.8)
TrainingSet3<-Pisos3[TrainIndex,] 
TestSet3<-Pisos3[-TrainIndex,]

modelo2_predictivo<- lm(Precio ~ Distrito + M2 + M2_2+Banos2+ Habitaciones + Ascensor + Terraza.Balcon +
                          Exterior.Interior + Banos + Estado+M2*Distrito+M2*Banos, data=TrainingSet3)

summary(modelo2_predictivo)

predicciones3<-predict(modelo2_predictivo, newdata = TestSet3)

# calculamos los errores de predicción 
errores3<-TestSet3$Precio-predicciones3
h<-nrow(TestSet3)

# calculo de las medidas de performance predictiva en el TestSet 
RMSE3<-sqrt(sum(errores3^2)/h)
RMSE3

#2.3. ARBOLES DE REGRESION y ARBOLES MODELO

str(PISOS_filtrado)

#partición 80%, 20%
RNGkind("Super", "Inversion")  
set.seed(123) 
n<-nrow(PISOS_filtrado)

TrainIndex_arbol<-sample(1:n, n*0.8) 
pisos_train<-PISOS_filtrado[TrainIndex_arbol,] 
pisos_test <-PISOS_filtrado[-TrainIndex_arbol,]

m.rpart <- rpart(Precio ~ ., data = pisos_train)
m.rpart
summary(m.rpart)

rpart.plot(m.rpart, digits = 3)

p.rpart <- predict(m.rpart, pisos_test)

summary(p.rpart)
summary(PISOS_filtrado$Precio)

errores4<-pisos_test$Precio-p.rpart
h<-nrow(pisos_test) 
RMSE4<-sqrt(sum(errores4^2)/h)
RMSE4

#Mejora del modelo
m.m5p <- M5P(Precio ~ ., data = pisos_train)
m.m5p
summary(m.m5p)

p.m5p <- predict(m.m5p, pisos_test)

summary(p.m5p)
summary(PISOS_filtrado$Precio)

errores5<-pisos_test$Precio-p.m5p
h<-nrow(pisos_test) 
RMSE5<-sqrt(sum(errores5^2)/h)
RMSE5

#2.4. REDES NEURONALES

Pisos_red_filtrado <- PISOS_filtrado 

str(Pisos_red_filtrado)

#particion
RNGkind("Super", "Inversion") 
set.seed(123)
n<-nrow(Pisos_red_filtrado) 
TrainIndex_red<-sample(1:n, n*0.8) 
train_red<-Pisos_red_filtrado [TrainIndex_red,] 
test_red <-Pisos_red_filtrado [-TrainIndex_red,]

#procesamos los datos para que esten en rango 0-1
preProcessRangeModel<-preProcess(train_red, method=c("range"))

trainproc<-predict(preProcessRangeModel, train_red)

summary(trainproc)
str(trainproc)

#parámetros de cross validation
control<-trainControl(method="repeatedcv", number=10, repeats=3)

#hiperparametros a optimizar con nnet
modelLookup("nnet")

#creamos grid de combinaciones de hiperparametros
grid<-expand.grid(size=c(2:12), decay=c(0, 0.001, 0.01))

#entrenamiento de la red
RNGkind("Super", "Inversion")
set.seed(123)

net<-train(Precio~., data=trainproc, method="nnet", trControl=control, tuneGrid=grid, metric = "RMSE")

net
plot(net)
net$bestTune

RMSEcv<-min(net$results[,3])
RMSEcv

#grafico de la red
plotnet(net, pos_col="green", neg_col="blue")

# importancia de las variables
garson(net)
varImp(net)

#predicciones en training set
predtrainproc<-predict(net, newdata=train_red)
RMSEtr<-RMSE(predtrainproc, trainproc$Precio)
RMSEtr

#predicciones en testset
#primero hay que preprocesar el test set
minprec<-min(test_red$Precio)
maxprec<-max(test_red$Precio)

testproc<-predict(preProcessRangeModel, test_red)

predtestproc<-predict(net, newdata=testproc)

RMSEtest<-RMSE(predtestproc, testproc$Precio)
RMSEtest

#deshacemos la transformacion de range para obtener predicciones en escala original
predtest<-predtestproc*(maxprec-minprec)+minprec
RMSEtest_original<-RMSE(predtest, test_red$Precio)
RMSEtest_original

#2.5. SUPPORT VECTOR MACHINE

Pisos_SVM_filtrado <- PISOS_filtrado 

str(Pisos_SVM_filtrado)

RNGkind("Super", "Inversion")  
set.seed(123)
n<-nrow(Pisos_SVM_filtrado)

#partición 80%, 20%
TrainIndex_SVM<-sample(1:n, n*0.8) 
pisos_train_SVM<-Pisos_SVM_filtrado[TrainIndex_SVM,] 
pisos_test_SVM <-Pisos_SVM_filtrado[-TrainIndex_SVM,]

#1. Kernel Lineal
svm <- ksvm(Precio ~ ., data = pisos_train_SVM,
             kernel = "vanilladot")

svm

pred <- predict(svm, newdata = pisos_test_SVM)
RMSEtest_SVM<-RMSE(pred, pisos_test_SVM$Precio)
RMSEtest_SVM 

#2. Kernel Gaussiano

svm2 <- ksvm(Precio ~ ., data = pisos_train_SVM,
            kernel = "rbfdot")

svm2

pred2 <- predict(svm2, newdata = pisos_test_SVM)
RMSEtest_SVM2<-RMSE(pred2, pisos_test_SVM$Precio)
RMSEtest_SVM2 #mejor resultado

#3. Kernel Polynominial

svm3 <- ksvm(Precio ~ ., data = pisos_train_SVM,
             kernel = "polydot")

svm3

pred3 <- predict(svm3, newdata = pisos_test_SVM)
RMSEtest_SVM3<-RMSE(pred3, pisos_test_SVM$Precio)
RMSEtest_SVM3

#2.6. ENSEMBLES

Pisos_Ensemble <- PISOS_filtrado 

RNGkind("Super", "Inversion")
set.seed(123)
n<-nrow(Pisos_Ensemble)

#partición 80%, 20%
TrainIndex_Ensemble<-sample(1:n, n*0.8) 
pisos_train_EN<-Pisos_Ensemble[TrainIndex_Ensemble,] 
pisos_test_EN <-Pisos_Ensemble[-TrainIndex_Ensemble,]

# Establecer parámetro de control.
control <- trainControl(method="repeatedcv", number=10, repeats=3,      
                        savePredictions=TRUE, search = "random")

# Establecer la lista de los algoritmos.
algorithmList <- c('rf','lm', 'svmRadial', "nnet")

# Entrenamos el conjunto de modelos.
set.seed(123)
models <- caretList(Precio ~., data=pisos_train_EN, 
                    metric="RMSE",
                    trControl=control, 
                    methodList=algorithmList)

#Resultados
results <- resamples(models)
summary(results)
dotplot(results)
modelCor(resamples(models))

#mejores resultados --> RF, LM

#STACK CON  RF COMO SECOND LEVEL ALGORITHM

stackControl <- trainControl(method="repeatedcv", 
                             number=10,
                             repeats=3, 
                             savePredictions=TRUE,
                             search="random")

stack.rf <- caretStack(models, method="rf",
                       metric="RMSE", 
                       trControl = stackControl)

print(stack.rf)

#STACK CON LM COMO SECOND LEVEL ALGORITHM

RNGkind("Super", "Inversion")
set.seed(321)
stack.lm <- caretStack(models, method="lm", metric="RMSE", trControl=stackControl)

print(stack.lm)

#PREDICCIONES DE STACKS

pred.RF<-predict(stack.rf , newdata=pisos_test_EN)
RMSEtest_EN_RF<-RMSE(pred.RF, pisos_test_EN$Precio) 
RMSEtest_EN_RF

pred.LM<-predict(stack.lm, newdata=pisos_test_EN)
RMSEtest_EN_LM<-RMSE(pred.LM, pisos_test_EN$Precio)
RMSEtest_EN_LM

##CONCLUSIONES ENSEMBLE --> mejor RMSE --> stacking con LM como second level algorithm.

#2.7. RANDOM FOREST

Pisos_RF <- PISOS_filtrado 
RNGkind("Super", "Inversion")
set.seed(123)
n<-nrow(Pisos_RF)

#partición 80%, 20%
TrainIndex_RF<-sample(1:n, n*0.8) 
pisos_train_RF<-Pisos_RF[TrainIndex_RF,] 
pisos_test_RF <-Pisos_RF[-TrainIndex_RF,]

#ver que hiperparametros pueden optimizarse en ranger
modelLookup("rf")

ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats =3)

grid_rf <- expand.grid(.mtry = c(2:20))

m_rf <- train(Precio ~ ., data = pisos_train_RF, method = "rf",
              metric = "RMSE", trControl = ctrl,
              tuneGrid = grid_rf)

m_rf
plot(m_rf)

varImp(m_rf)

#prediccion

predRF<-predict(m_rf, newdata=pisos_test_RF)
RMSEtest_RF<-RMSE(predRF, pisos_test_RF$Precio)
RMSEtest_RF


