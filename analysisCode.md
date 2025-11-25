```R
####差异分析
library(limma)
library(dplyr)
df<-read.csv("exp.csv",header=T,row.names=1,check.names=F)
list<-c(rep("Tumor",151),rep("Control",1142)) %>% factor(., levels=c("Tumor","Control"),ordered=F)
list
list <- model.matrix(~factor(list)+0)  #把group设置成一个model matrix
colnames(list) <- c("Tumor", "Control")
list
df.fit <- lmFit(df, list)
df.matrix <- makeContrasts(Tumor - Control,levels=list)
fit <- contrasts.fit(df.fit, df.matrix)
fit <- eBayes(fit)
tempOutput <- topTable(fit,n = Inf, adjust = "fdr")
head(tempOutput)
DEG<-tempOutput
DEG$regulate <- ifelse(DEG$P.Value > 0.05, "unchanged",ifelse(DEG$logFC > 0, "up-regulated",ifelse(DEG$logFC < 0, "down-regulated", "unchanged")))

## 导出所有的差异结果

nrDEG = na.omit(DEG) ## 去掉数据中有NA的行或列
write.csv(nrDEG, "all.limmaOut.csv",quote=F)

####GSVA评分
exp<-as.matrix(exp)
#读取基因集
#第一列为基因名，第二列为类型
gene_set <- read.table("./mmc3.txt", header = T, sep ="\t")
geneSet <- split(as.matrix(gene_set)[,2], gene_set[,1])
gsva_param <- gsvaParam(as.matrix(train_data), geneSet, maxDiff=TRUE, kcdf="Gaussian")
gsva_results <- gsva(gsva_param)
res <- gsva_results %>% t() %>% as.data.frame()


####KM生存分析
#OS为整合后的生存数据
diff=survdiff(Surv(time, status)~Group,data=OS)
pValue=diff$p
if(pValue<0.001){
pValue="P<0.001"
}else{ pValue=paste0("P=",sprintf("%.03f",pValue))
}

fit <- survfit(Surv(time, status) ~Group, data = OS)
surPlot=ggsurvplot(fit,
surv.median.line = "hv",
data=OS,
conf.int=T, #置信区间
pval=pValue,
pval.size=4,
legend.labs=c("HighRisk","LowRisk"),
legend.title="Group",
palette = c("#fa6d1d", "#0780cf"),
xlab="Time(Month)",
break.time.by = 12,
risk.table=T,
risk.table.height=.25)



####整合机器学习
work.path <- getwd(); setwd(work.path) 

# 设置其他路径

code.path <- file.path(work.path, "Codes") # 存放脚本
data.path <- file.path(work.path, "InputData") # 存在输入数据（需用户修改）
res.path <- file.path(work.path, "Results") # 存放输出结果
fig.path <- file.path(work.path, "Figures") # 存放输出图片

# 如不存在这些路径则创建路径

if (!dir.exists(data.path)) dir.create(data.path)
if (!dir.exists(res.path)) dir.create(res.path)
if (!dir.exists(fig.path)) dir.create(fig.path)
if (!dir.exists(code.path)) dir.create(code.path)

# BiocManager::install("mixOmics")

# BiocManager::install("survcomp")

# devtools::install_github("binderh/CoxBoost")

# install.packages("randomForestSRC")

# install.packages("snowfall")

# 加载需要使用的R包

library(openxlsx)
library(seqinr)
library(plyr)
library(survival)
library(randomForestSRC)
library(glmnet)
library(plsRcox)
library(superpc)
library(gbm)
library(mixOmics)
library(survcomp)
library(CoxBoost)
library(survivalsvm)
library(BART)
library(snowfall)
library(ComplexHeatmap)
library(RColorBrewer)
source(file.path(code.path, "ML.R"))
FinalModel <- c("panML", "multiCox")[2]

# 训练集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与测试集保持相同类型，表达谱需有一定变异性，以免建模过程报错）

Train_expr<- read.table(file.path(data.path, "Training_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
#Train_expr<-Train_expr+1

# 训练集生存数据是行为样本，列为结局信息的数据框（请确保生存时间均大于0）

Train_surv <- read.table(file.path(data.path, "Training_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
#剔除OS.time=0的样本
Train_surv<-Train_surv[which(Train_surv$OS.time>0),]
comsam <- intersect(rownames(Train_surv), colnames(Train_expr))
Train_expr <- Train_expr[,comsam]; Train_surv <- Train_surv[comsam,,drop = F]
Train_expr<-Train_expr[gene,]

# 测试集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与训练集保持相同类型）

Test_expr <- read.table(file.path(data.path, "Testing_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
#Test_expr<-Test_expr+1

# 测试集生存数据是行为样本，列为结局信息的数据框（请确保生存时间均大于0）

Test_surv <- read.table(file.path(data.path, "Testing_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Test_surv<-Test_surv[which(Test_surv$OS.time>0),]
comsam <- intersect(rownames(Test_surv), colnames(Test_expr))
Test_expr <- Test_expr[,comsam]; Test_surv <- Test_surv[comsam,,drop = F]
Test_expr<-Test_expr[gene,]

# 提取相同基因

comgene <- intersect(rownames(Train_expr),rownames(Test_expr))
Train_expr <- t(Train_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因
Test_expr <- t(Test_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因
Train_set = scaleData(data = Train_expr, centerFlags = F, scaleFlags = T) 
names(x = split(as.data.frame(Test_expr), f = Test_surv$Cohort)) # 注意测试集标准化顺序与此一致
Test_set = scaleData(data = Test_expr, cohort = Test_surv$Cohort, centerFlags = F, scaleFlags = T)
methods <- read.xlsx(file.path(code.path, "41467_2022_28421_MOESM4_ESM.xlsx"), startRow = 2)$Model
methods <- gsub("-| ", "", methods)

## Train the model --------------------------------------------------------

min.selected.var <- 2 # 筛选变量数目的最小阈值
timeVar = "OS.time"; statusVar = "OS" # 定义需要考虑的结局事件，必须出现在Train_surv以及Test_surv中

## Pre-training 

Variable = colnames(Train_expr)
preTrain.method =  strsplit(methods, "\\+")
preTrain.method = lapply(preTrain.method, function(x) rev(x)[-1])
preTrain.method = unique(unlist(preTrain.method))
preTrain.method

set.seed(seed = 2024) # 设置建模种子，使得结果可重复
preTrain.var <- list()
for (method in preTrain.method){
  preTrain.var[[method]] = RunML(method = method, # 机器学习方法
                                 Train_expr = Train_set, # 训练集有潜在预测价值的变量
                                 Train_surv = Train_surv, # 训练集生存数据
                                 mode = "Variable",       # 运行模式，Variable(筛选变量)和Model(获取模型)
                                 classVar = classVar) # 用于训练的生存变量，必须出现在Train_surv中
}
preTrain.var[["simple"]] <- colnames(Train_expr)

model <- list() # 初始化模型结果列表
set.seed(seed = 123) # 设置建模种子，使得结果可重复
for (method in methods){ # 循环每一种方法组合

  # method <- "CoxBoost+plsRcox" # [举例]若遇到报错，请勿直接重头运行，可给method赋值为当前报错的算法来debug

  cat(match(method, methods), ":", method, "\n") # 输出当前方法
  method_name = method # 本轮算法名称
  method <- strsplit(method, "\\+")[[1]] # 各步骤算法名称

  if (length(method) == 1) method <- c("simple", method)

  selected.var = preTrain.var[[method[1]]]

  # 如果筛选出的变量小于阈值，则该算法组合无意义，置空（尤其针对以RSF筛选变量的情况，需在ML脚本中尝试调参）

  if (length(selected.var) <= min.selected.var) {
    model[[method_name]] <- NULL
  } else {
    model[[method_name]] <- RunML(method = method[2], # 用于构建最终模型的机器学习方法
                                  Train_expr = Train_expr[, selected.var], # 训练集有潜在预测价值的变量
                                  Train_surv = Train_surv, # 训练集生存数据
                                  mode = "Model",       # 运行模式，Variable(筛选变量)和Model(获取模型)
                                  classVar = classVar)  # 用于训练的生存变量，必须出现在Train_surv中
  }

  # 如果最终筛选出的变量小于阈值，则该算法组合也无意义，置空

  if(length(ExtractVar(model[[method_name]])) <= min.selected.var) {
    model[[method_name]] <- NULL
  }
}
saveRDS(model, file.path(res.path, "model.rds")) # 保存所有模型输出

# 当要求最终模型为多变量cox时，对模型进行更新

if (FinalModel == "multiCox"){
  coxmodel <- lapply(model, function(fit){ # 根据各算法最终获得的变量，构建多变量cox模型，从而以cox回归系数和特征表达计算单样本风险得分
    tmp <- coxph(formula = Surv(Train_surv[[timeVar]], Train_surv[[statusVar]]) ~ .,
                 data = as.data.frame(Train_set[, ExtractVar(fit)]))
    tmp$subFeature <- ExtractVar(fit) # 2.1版本更新，提取当B模型依旧降维情况下的最终变量
    return(tmp)
  })
}
saveRDS(coxmodel, file.path(res.path, "coxmodel.rds")) # 保存最终以多变量cox拟合所筛选变量的模型

## Evaluate the model -----------------------------------------------------

# 读取已保存的模型列表（请根据需要调整）

model <- readRDS(file.path(res.path, "model.rds")) # 若希望使用各自模型的线性组合函数计算得分，请运行此行

# model <- readRDS(file.path(res.path, "coxmodel.rds")) # 若希望使用多变量cox模型计算得分，请运行此行

methodsValid <- names(model) # 取出有效的模型（变量数目小于阈值的模型视为无效）

# 根据给定表达量计算样本风险评分

RS_list <- list()
for (method in methodsValid){
  RS_list[[method]] <- CalRiskScore(fit = model[[method]], 
                                    new_data = rbind.data.frame(Train_set,Test_set), # 4.0更新
                                    type = "lp") # 同原文，使用linear Predictor计算得分

}
RS_mat <- as.data.frame(t(do.call(rbind, RS_list)))
write.table(RS_mat, file.path(res.path, "RS_mat.txt"),sep = "\t", row.names = T, col.names = NA, quote = F) # 输出风险评分文件

# 提取所筛选的变量（列表格式）

fea_list <- list()
for (method in methodsValid) {
  fea_list[[method]] <- ExtractVar(model[[method]]) # 2.1版本更新，提取当B模型依旧降维情况下的最终变量
}

# 提取所筛选的变量（数据框格式）

fea_df <- lapply(model, function(fit){ data.frame(ExtractVar(fit)) }) # 2.1版本更新，提取当B模型依旧降维情况下的最终变量
fea_df <- do.call(rbind, fea_df)
fea_df$algorithm <- gsub("(.+)\\.(.+$)", "\\1", rownames(fea_df))
colnames(fea_df)[1] <- "features"  # 数据框有两列，包含算法以及算法所筛选出的变量
write.table(fea_df, file.path(res.path, "fea_df.txt"),sep = "\t", row.names = F, col.names = T, quote = F)

# 对各模型计算C-index

Cindexlist <- list()
for (method in methodsValid){
  Cindexlist[[method]] <- RunEval(fit = model[[method]], # 预后模型
                                  Test_expr = Test_set, # 测试集预后变量，应当包含训练集中所有的变量，否则会报错
                                  Test_surv = Test_surv, # 训练集生存数据，应当包含训练集中所有的变量，否则会报错
                                  Train_expr = Train_set, # 若需要同时评估训练集，则给出训练集表达谱，否则置NULL
                                  Train_surv = Train_surv, # 若需要同时评估训练集，则给出训练集生存数据，否则置NULL
                                  Train_name = "TCGA-GBM", # 若需要同时评估训练集，可给出训练集的标签，否则按“Training”处理
                                  #Train_expr = NULL,
                                  #Train_surv = NULL, 
                                  cohortVar = "Cohort", # 重要：用于指定队列的变量，该列必须存在且指定[默认为“Cohort”]，否则会报错
                                  timeVar = timeVar, # 用于评估的生存时间，必须出现在Test_surv中；这里是OS.time
                                  statusVar = statusVar) # 用于评估的生存状态，必须出现在Test_surv中；这里是OS
}
Cindex_mat <- do.call(rbind, Cindexlist)
write.table(Cindex_mat, file.path(res.path, "cindex_mat.txt"),sep = "\t", row.names = T, col.names = T, quote = F)

# Plot --------------------------------------------------------------------

Cindex_mat <- read.table(file.path(res.path, "cindex_mat.txt"),sep = "\t", row.names = 1, header = T,check.names = F,stringsAsFactors = F)
avg_Cindex <- sort(apply(Cindex_mat, 1, mean), decreasing = T) # 计算每种算法在所有队列中平均C-index，并降序排列
Cindex_mat <- Cindex_mat[names(avg_Cindex), ] # 对C-index矩阵排序
avg_Cindex <- as.numeric(format(avg_Cindex, digits = 3, nsmall = 3)) # 保留三位小数
fea_sel <- fea_list[[rownames(Cindex_mat)[1]]] # 最优模型（即测试集[或者训练集+测试集]C指数均值最大）所筛选的特征

CohortCol <- brewer.pal(n = ncol(Cindex_mat), name = "Paired") # 设置绘图时的队列颜色
names(CohortCol) <- colnames(Cindex_mat)

# 调用简易绘图函数

cellwidth = 1; cellheight = 0.5
hm <- SimpleHeatmap(Cindex_mat = Cindex_mat, # 主矩阵
                    avg_Cindex = avg_Cindex, # 侧边柱状图
                    CohortCol = CohortCol, # 列标签颜色
                    barCol = "steelblue", # 右侧柱状图颜色
                    col = c("#1CB8B2", "#FFFFFF", "#EEB849"), # 热图颜色
                    cellwidth = cellwidth, cellheight = cellheight, # 热图每个色块的尺寸
                    cluster_columns = F, cluster_rows = F) # 是否对行列进行聚类

pdf(file.path(fig.path, "heatmap of cindex.pdf"), width = cellwidth * ncol(Cindex_mat) + 4, height = cellheight * nrow(Cindex_mat) * 0.45)
draw(hm, heatmap_legend_side = "right", annotation_legend_side = "right") # 热图注释均放在右侧
invisible(dev.off())


####临床特征C指数对比

# 设置工作路径

work.path <- getwd(); setwd(work.path) 

# 设置其他路径

code.path <- file.path(work.path, "Codes") # 存放脚本
data.path <- file.path(work.path, "InputData") # 存在输入数据（需用户修改）
res.path <- file.path(work.path, "Results") # 存放输出结果
fig.path <- file.path(work.path, "Figures") # 存放输出图片

# 如不存在这些路径则创建路径

if (!dir.exists(data.path)) dir.create(data.path)
if (!dir.exists(res.path)) dir.create(res.path)
if (!dir.exists(fig.path)) dir.create(fig.path)
if (!dir.exists(code.path)) dir.create(code.path)

# 加载R包

library(org.Hs.eg.db)
library(survival)
library(ggplot2)
library(cowplot)
library(RColorBrewer)
library(openxlsx)

# 加载用于模型比较的脚本

source(file.path(code.path, "compare.R"))

# 读取训练集和测试集 -----------------------------------------------------------

## Training Cohort -------------------------------------------------------------

# 训练集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与测试集保持相同类型，表达谱需有一定变异性，以免建模过程报错）

Train_expr <- read.table(file.path(data.path, "Training_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)

# 训练集生存数据是行为样本，列为结局信息的数据框（请确保生存时间均大于0）

Train_surv <- read.table(file.path(data.path, "Training_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
rownames(Train_surv)<-paste0(rownames(Train_surv),"-01")
comsam <- intersect(rownames(Train_surv), colnames(Train_expr))
Train_expr <- Train_expr[,comsam]; Train_surv <- Train_surv[comsam,,drop = F]

## Validation Cohort -----------------------------------------------------------

# 测试集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与训练集保持相同类型）

Test_expr <- read.table(file.path(data.path, "Testing_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)

# 测试集生存数据是行为样本，列为结局信息的数据框（请确保生存时间均大于0）

Test_surv <- read.table(file.path(data.path, "Testing_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
comsam <- intersect(rownames(Test_surv), colnames(Test_expr))
Test_expr <- Test_expr[,comsam]; Test_surv <- Test_surv[comsam,,drop = F]

# 提取相同基因

comgene <- intersect(rownames(Train_expr),rownames(Test_expr))
Train_expr <- t(Train_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因
Test_expr <- t(Test_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因

# 按队列对数据分别进行标准化（根据情况调整centerFlags和scaleFlags）

## data: 需要表达谱数据（行为样本，列为基因） 

## cohort：样本所属队列，为向量，不输入值时默认全表达矩阵来自同一队列

## centerFlag/scaleFlags：是否将基因均值/标准差标准化为1；

##        默认参数为NULL，表示不进行标准化；

##        为T/F时，表示对所有队列都进行/不进行标准化

##        输入由T/F组成的向量时，按顺序对队列进行处理，向量长度应与队列数一样

##        如centerFlags = c(F, F, F, T, T)，表示对第4、5个队列进行标准化，此时flag顺序应当与队列顺序一致

##        如centerFlags = c("A" = F, "C" = T, "B" = F)，表示对队列C进行标准化，此时不要求flag顺序与data一致

Train_set = scaleData(data = Train_expr, centerFlags = F, scaleFlags = F) 
names(x = split(as.data.frame(Test_expr), f = Test_surv$Cohort)) # 注意测试集标准化顺序与此一致
Test_set = scaleData(data = Test_expr, cohort = Test_surv$Cohort, centerFlags = F, scaleFlags = F)

## Public Signature ------------------------------------------------------------

## pubSIG2（该文件需由用户仿照示例文件public signatures.txt提供）

# 文件为txt格式；且至少2或3列信息，具体如下：

# 必须列“Model”：以区别不同签名

# 必须列“SYMBOL”：表示签名中所含的基因

# 可选列“Coef”：代表已知签名的系数，若用户未提供该列则根据多变量Cox计算每个基因的系数

pubSIG <- read.table(file.path(data.path, "public signatures.txt"), header = T)
if (!"Coef" %in% colnames(pubSIG)) pubSIG$Coef <- NA # 若未匹配到“Coef“列，则新建“Coef“列并默认为NA
pubSIG <- split(pubSIG[, c("SYMBOL", "Coef")], pubSIG$Model)

## My Signature ----------------------------------------------------------------

mySIGname = "GBMRS" # 本研究所定义的签名的名字，用于在图形中显示
myAlgorithm = "CoxBoost+RSF" # 本研究所定义的最优算法，即热图最顶部的算法名称

## mySIG1：RS_mat，使用PrognosticML脚本生成的风险评分文件，此评分是机器学习算法通过predict函数直接获取

mySIG <- read.table(file.path(res.path, "RS_mat.txt"), header = T, check.names = F,sep="\t",row.names=1)
mySIG <- setNames(object = mySIG[[myAlgorithm]], nm = rownames(mySIG))
signatures <- pubSIG
signatures[[mySIGname]] <- mySIG

## 计算C指数 -------------------------------------------------------------------

model <- list(); cinfo <- list() # 初始化变量
log.file <- file.path(res.path, "makeCox.log") # 在Results文件夹下新建log文件
if (file.exists(log.file)) file.remove(log.file) # 此log文件用于存放在进行多变量cox分析时的警告
log.file <- file(log.file, open = "a")
sink(log.file, append = TRUE, type = "message")
for (i in names(signatures)){
  if (class(signatures[[i]]) == "data.frame"){
    model[[i]] <- makeCox(Features = signatures[[i]]$SYMBOL, # 签名的基因名
                          coefs = signatures[[i]]$Coef,      # 公共签名所提供的基因系数（如未提供也不必修改此行代码）
                          SIGname = i,                       # 当前循环的签名
                          unmatchR = 0.3,                    # 基因名不匹配率，高于该比率将被剔除；低于匹配率但大于0时会报警告，并存入log文件
                          Train_expr = Train_set,            # 用于计算cox系数的训练集表达谱
                          Train_surv = Train_surv,           # 用于计算cox系数的训练集生存信息
                          statusVar = "OS",                  # 用于构建cox模型的生存结局
                          timeVar = "OS.time")               # 用于构建cox模型的生存时间
  }else{
    model[[i]] = signatures[[i]]
  }

  cinfo[[i]] <- calCindex(model = model[[i]],                # 训练的cox模型，为有名字的向量
                          name = i,                          # 当前循环的签名
                          Test_expr = Test_set,              # 用于计算c指数的测试集表达谱
                          Test_surv = Test_surv,             # 用于计算c指数的测试集生存信息
                          Train_expr = Train_set,            # 用于计算c指数的训练集表达谱
                          Train_surv = Train_surv,           # 用于计算c指数的训练集生存信息
                          Train_name = "TCGA-GBM",               # 指定训练集的名称
                          #Train_expr = NULL,                # 若不需要评估训练集，则取消此行注释，并注释掉上方对应行
                          #Train_surv = NULL,                # 若不需要评估训练集，则取消此行注释，并注释掉上方对应行
                          CohortVar = "Cohort",              # 用于指定测试集所来自的队列
                          metaCohort = FALSE,                 # 指示是否将测试集合并生成MetaCohort
                          statusVar = "OS",                  # 用于计算c指数的生存结局
                          timeVar = "OS.time")               # 用于计算c指数的生存时间
  message("")
}
closeAllConnections()

cinfo <- do.call(rbind, cinfo)
write.table(cinfo[,1:5], file = file.path(res.path,"cinfo.txt"),sep = "\t",row.names = T,col.names = NA,quote = F) # 输出不同签名在所有队列中的c指数统计量
cinfo <- split(cinfo, cinfo$Cohort)

# 绘图 -------------------------------------------------------------------------

CohortCol <- brewer.pal(n = length(cinfo), name = "Paired") # 设置绘图时的队列颜色
names(CohortCol) <- names(cinfo)

# 批量绘制各个队列的森林图

plots <- lapply(cinfo, function(plot.data){
  plot.data$method <- 
    factor(plot.data$method,
           levels = plot.data$method[order(plot.data$C, decreasing = F)])

  # compares two concordance indices: the statistical test is a two-sided Student t test for dependent samples.

  C.compare <- plot.data$C[plot.data$method]
  se.compare <- plot.data$se[plot.data$method]
  n.compare <- plot.data$n[plot.data$method]
  RS.compare <- plot.data$RS[plot.data$method][[1]]
  r.combined <- unlist(lapply(plot.data$RS, function(x) cor(x, RS.compare)))
  var.combined <- plot.data$se^2 + se.compare^2 - 2*r.combined*plot.data$se*se.compare
  p <- pt(abs((plot.data$C-C.compare))/(sqrt(var.combined)), n.compare - 1, lower.tail = F) * 2
  plot.data$label <- cut(p, breaks = c(0, 0.05, 0.01, 0.001, 0.0001))
  plot.data$label <- plyr::mapvalues(x = plot.data$label,
                                     from = c("(0,0.0001]", "(0.0001,0.001]", "(0.001,0.01]", "(0.01,0.05]"), 
                                     to = c("****", "***", "**", "*"))

  return(ggplot(plot.data, aes(x = method, y = C, fill = Cohort)) +
           geom_errorbar(aes(ymin = C - 1.96 * se, ymax = C + 1.96 * se), width = .1) +
           geom_point(color = CohortCol[unique(plot.data$Cohort)], size = 2.5) +
           geom_text(aes(x = method, y = max(plot.data$C + 1.96 * plot.data$se - 0.05), label = label)) +
           geom_hline(yintercept = 0.6, linetype = "dashed") +
           ggtitle(label = unique(plot.data$Cohort)) +
           coord_flip() + 
           theme_classic() +
           theme(panel.border = element_rect(fill = NA, size = 1),
                 axis.title = element_blank(),
                 legend.position = "none"))
})

# 森林图合并并保存

plot_grid(plotlist = plots, nrow = 1)
ggsave(file.path(fig.path, "comparison.pdf"), width = 15, height = 10)



#单因素多因素
vars<-colnames(df)[-c(1:2)]#自变量
sur<-Surv(time=df$OS,event=as.numeric(df$Status))
UniCox<-function(x){
   FML<-as.formula(paste0('sur~',x))
   fit<-coxph(FML,data =df)
   fitSum<-summary(fit)
   HR<-round(fitSum$coefficients[,2],3)
   PValue<-ifelse(fitSum$coefficients[,5]<0.001,format(fitSum$coefficients[,5],scientific=T),round(fitSum$coefficients[,5],3))
   Lower<-round(fitSum$conf.int[,3],3)
   Upper<-round(fitSum$conf.int[,4],3)
   Unicox<-data.frame('Characteristics'=x,
		'Hazard Ratio'=HR,
		'lower.95'=Lower,
		'upper.95'=Upper,
		'CI95%'=paste(HR,"[",Lower,":",Upper,"]",sep=""),
		'P Value'=PValue)
   return(Unicox)
}
UniVar<- lapply(vars,UniCox)
UniVar<-ldply(UniVar,data.frame)
UniVar
UniVar$Characteristics[UniVar$P.Value<0.05]

multSeed<-as.formula(sur~.)
MultCox<-coxph(multSeed,data=df)
MultSum<-summary(MultCox)
multi1<-as.data.frame(round(MultSum$conf.int[, c(1, 3, 4)], 3))
multi2<-ShowRegTable(MultCox, 
                     exp=TRUE, 
                     digits=3, 
                     pDigits =3,
                     printToggle = TRUE, 
                     quote=FALSE, 
                     ciFun=confint)
resultMul <-cbind(multi1,multi2)
resultMul<-tibble::rownames_to_column(resultMul, var = "Characteristics")
resultMul

#列线图
library(rms)
library(survival)
dd<-datadist(data)
norm<-cph(Surv(OS,Status)~.,data=df,x=TRUE,y=TRUE,surv=TRUE)
#绘制列线图
surv<-Survival(f)
surv1<-function(x)surv(365,x)
surv2<-function(x)surv(1095,x)
surv3<-function(x)surv(1825,x)

nom<-nomogram(f,fun=list(surv1,surv2,surv3),lp=F,fun.at=c(0.05,seq(0.2,0.8,by=0.1),0.95),funlabel=c('1 year survival','3 year survival','5 year survival'))
pdf("nomogram.pdf",width=10)
plot(nom,xfrac = 0.2,tcl=-0.2,lmgp=0.1,cex.var=1,cex.axis=1,col.grid=gray(c(0.8,0.95)))

dev.off()

#校准曲线
f1<-cph(Surv(OS,Event)~.,data=dd,x=TRUE,y=TRUE,surv=TRUE,time.inc=1*365)
f3<-cph(Surv(OS,Event)~.,data=dd,x=TRUE,y=TRUE,surv=TRUE,time.inc=3*365)
f5<-cph(Surv(OS,Event)~.,data=dd,x=TRUE,y=TRUE,surv=TRUE,time.inc=5*365)
cali_1<-calibrate(f1,cmethod="KM",method="boot",u=1*365,m=100,B=1000)
cali_3<-calibrate(f3,cmethod="KM",method="boot",u=3*365,m=100,B=1000)
cali_5<-calibrate(f5,cmethod="KM",method="boot",u=5*365,m=100,B=1000)
plot(cali_1,lwd=1,lty=1,errbar.col="black",xlim=c(0.2,1),ylim=c(0.2,1),xlab="Nomogram-Predicted Probability of 1,3,5 Year Survival",ylab="Actual 1,3,5 Year Survival",col="#FF6347",sub=F)
plot(cali_3,add=T,lwd=1,lty=1,errbar.col="black",col="#6A5ACD",sub=F)
plot(cali_5,add=T,lwd=1,lty=1,errbar.col="black",col="#32CD32",sub=F)
legend("bottomright",legend=c("1-year","3-year","5-year"),col=c("#FF6347","#6A5ACD","#32CD32"),lwd=2,lty=1)

#决策曲线
library("ggDCA")
dca_cph<-dca(f,model.names=c("Nomogram"),times=c(365,1095,1825))
#Error in Surv(data, OS) : Time variable is not numeric
#上述报错需要f<-cph(Surv(OS,Event)构建模型得时候直接用OS和Status,不用data$OS
ggplot(dca_cph)
library(ggsci)
ggplot(dca_cph,linetype = F)+scale_color_jama(name="Model Type")+
  theme_bw()+
  facet_wrap(~time)
```

