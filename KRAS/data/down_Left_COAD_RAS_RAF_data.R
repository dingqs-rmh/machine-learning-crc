setwd("~/project/KRAS/data")
rm(list = ls())

group1 <- read.table("COAD_KRAS_type.txt", header = T, sep = '\t')
group2 <- read.table("COAD_BRAF_type.txt", header = T, sep = '\t')
group3 <- read.table("COAD_NRAS_type.txt", header = T, sep = '\t')
# 取并集
union_samples <- unique(c(group1$Sample, group2$Sample, group3$Sample))

# 创建新的数据框
group <- data.frame(Sample = union_samples)
#group <- read.table("COAD_KRAS_type.txt", header = T, sep = '\t')



library(TCGAbiolinks)
library(dplyr)
library(stringr)
library(SummarizedExperiment)
library(org.Hs.eg.db)
options(scipen = 999)

# 下载TCGA数据集，例如TCGA-COAD
query <- GDCquery(project = "TCGA-COAD", 
                  data.category = "Transcriptome Profiling", 
                  data.type = "Gene Expression Quantification", 
                  workflow.type = "STAR - Counts")

GDCdownload(query, method = "api", files.per.chunk = 20)

data <- GDCprepare(query)

tpm_data <- assay(data, "tpm_unstrand")
metadata <- colData(data)

# 确保基因ID没有版本号（移除点号及其后面的数字）
cleaned_ensembl_ids <- sub("\\..*", "", rownames(tpm_data))

# 使用biomaRt来获取基因的类型信息
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

# 获取基因的类型（biotype）信息
gene_annotations <- getBM(attributes = c("ensembl_gene_id", "gene_biotype"),
                          filters = "ensembl_gene_id",
                          values = cleaned_ensembl_ids,
                          mart = mart)

# 只保留protein_coding基因
protein_coding_genes <- gene_annotations[gene_annotations$gene_biotype == "protein_coding", "ensembl_gene_id"]

# 检查有多少protein_coding基因
cat("Number of protein_coding genes: ", length(protein_coding_genes), "\n")

# 过滤tpm_data数据集，只保留protein_coding基因
tpm_data <- tpm_data[cleaned_ensembl_ids %in% protein_coding_genes, ]

# 再次检查数据是否为空
if (nrow(tpm_data) == 0) {
  stop("No protein_coding genes found in the dataset.")
}

# 使用清理后的ID进行映射
mapped_genes <- mapIds(org.Hs.eg.db, 
                       keys = sub("\\..*", "", rownames(tpm_data)),
                       column = "SYMBOL",
                       keytype = "ENSEMBL",
                       multiVals = "first")

# 将映射后的基因符号作为行名
rownames(tpm_data) <- mapped_genes

# 去掉NA的行
tpm_data <- tpm_data[!is.na(rownames(tpm_data)), ]
# 简化样本 ID 并提取样本类型（01、11 等）
sample_types <- substr(colnames(tpm_data), 14, 16)

# 只保留 01 类型样本（Primary Tumor）
tpm_data <- tpm_data[, sample_types == "01A"]
# 简化列名
simplified_sample_ids <- sub("^([A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+-[0-9]{2}).*", "\\1", colnames(tpm_data))

# 检测并保留最新列（出现重复时保留最后出现的列）
duplicated_indices <- duplicated(simplified_sample_ids, fromLast = TRUE)

# 去掉重复的列，保留最后出现的列
tpm_data <- tpm_data[, !duplicated_indices]

# 更新列名
colnames(tpm_data) <- simplified_sample_ids[!duplicated_indices]

# 转置数据
transposed_tpm_data <- t(tpm_data)

# 将基因名称作为列名
colnames(transposed_tpm_data) <- rownames(tpm_data)

# 将简化的样本ID作为行名
rownames(transposed_tpm_data) <- simplified_sample_ids[!duplicated_indices]



# 下载和解析临床数据
clinical_query <- GDCquery(
  project = "TCGA-COAD",
  data.category = "Clinical",
  data.type = "Clinical Supplement",
  data.format = "BCR XML"
)
GDCdownload(clinical_query)
clinical_data <- GDCprepare_clinic(clinical_query, clinical.info = "patient")

# 筛选左侧结肠的样本
left_colon_samples <- clinical_data %>%
  filter(anatomic_neoplasm_subdivision %in% c("Descending Colon", "Sigmoid Colon", "Splenic Flexure")) %>%
  pull(bcr_patient_barcode)

# 简化样本ID格式，确保与临床数据格式一致
simplified_sample_ids <- sub("^([A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+)-[0-9]{2}.*", "\\1", colnames(tpm_data))

# 筛选tpm_data中左侧结肠的样本
left_colon_tpm_data <- tpm_data[, simplified_sample_ids %in% left_colon_samples]

# 检查筛选后的数据是否为空
if (ncol(left_colon_tpm_data) == 0) {
  stop("No left-side colon cancer samples found in the expression data.")
}

# 转置数据，使基因为列名，样本为行名
transposed_left_colon_tpm_data <- t(left_colon_tpm_data)

# 更新列名和行名
colnames(transposed_left_colon_tpm_data) <- rownames(left_colon_tpm_data)
rownames(transposed_left_colon_tpm_data) <- simplified_sample_ids[simplified_sample_ids %in% left_colon_samples]



# 接下来的代码
#group$Sample <- paste0(group$Sample, "-01")
Target <- ifelse(rownames(transposed_left_colon_tpm_data) %in% group$Sample, 1, 0)

# 将Target列移到首列
final_tpm_data <- cbind(Target, transposed_left_colon_tpm_data)

# 查看结果
head(final_tpm_data)

# 保存为CSV文件
write.csv(final_tpm_data, "LEFT_COAD_RAS_RAF_with_SampleID.csv", row.names = TRUE, quote = F)

