
#For test data
setwd('~/A_Projects/BiGR/')
library(limma)
library(TCGAbiolinks)
library(SummarizedExperiment)

#This commented code chunk will download the entierty of the
#TCGA database in regards to gene expression ~4.6 Gb as of 02/27/2023

# TCGA_code_vect_full<-c('BRCA','OV','LUAD','UCEC','GBM',
#                   'HNSC','KIRC','LGG','LUSC','THCA',
#                   'PRAD','SKCM','COAD','STAD','BLCA',
#                   'LIHC','CESC','KIRP','SARC','ESCA',
#                   'PAAD','READ','PCPG','TGCT','LAML',
#                   'THYM','ACC','MESO','UVM','KICH',
#                   'UCS','CHOL','DLBC')
# TCGA_code_vect_full<-paste0('TCGA-',TCGA_code_vect_full)
# #Will automatically download TCGA data
# #Data will not be downloaded if it is already present on the computer
# query <- GDCquery(
#   project = TCGA_code_vect_full,
#   data.category = "Gene expression",
#   data.type = "Gene expression quantification",
#   platform = "Illumina HiSeq", 
#   file.type  = "normalized_results",
#   experimental.strategy = "RNA-Seq",
#   legacy = TRUE
# )
# GDCdownload(
#   query = query, 
#   method = "api", 
#   files.per.chunk = 10
# )
# 
# TCGA_obj<-GDCprepare(query = query)
# 
# save(TCGA_obj,file='~/A_Projects/BiGR/TCGA_all_dta.rda')

load('~/A_Projects/BiGR/TCGA_all_dta_preprocessed.rda')

# quantile filter of genes
dataFilt <- TCGAanalyze_Filtering(
  tabDF = TCGA_obj_processed,
  method = "quantile", 
  qnt.cut =  0.25
)
rm(TCGA_obj_processed)

# selection of normal samples "NT"
samplesNT <- TCGAquery_SampleTypes(
  barcode = colnames(dataFilt),
  typesample = c("NT")
)

# selection of tumor samples "TP"
samplesTP <- TCGAquery_SampleTypes(
  barcode = colnames(dataFilt), 
  typesample = c("TP")
)

#Set-up matrices for DEA
mat1 = dataFilt[,samplesNT]
mat2 = dataFilt[,samplesTP]
TOC <- cbind(mat1, mat2)
Cond1num <- ncol(mat1)
Cond2num <- ncol(mat2)


#Extract patient IDs for duplicate correlation function of limma
Patients <- factor(my_IDs$patient)


#Prepare sample data for design and contrasts
Cond1type = "Normal"
Cond2type = "Tumor"

message("o ",Cond1num," samples in Cond1type ",Cond1type)
message("o ",Cond2num," samples in Cond2type ",Cond2type)
message("o ", nrow(TOC), " features as miRNA or genes ")

colnames(TOC) <- paste0("s", 1:ncol(TOC))

tumorType <- factor(
  x = rep(c(Cond1type, Cond2type), c(Cond1num, Cond2num)),
  levels = c(Cond1type, Cond2type)
)

#Design
design <- model.matrix(~ 0 + tumorType)

#Voom
message("Voom Transformation...")
logCPM <- limma::voom(TOC, design)

#Prep for contrast
colnames(design)[1:2] <- c(Cond1type, Cond2type)
contr <- paste0(Cond2type, "-", Cond1type)

#Create constrasts and fit
cont.matrix <- limma::makeContrasts(contrasts = contr, levels = design)
message("lmFit...")
fit <- limma::lmFit(logCPM, design)
rm(logCPM)

#Create contrasts
fit <- limma::contrasts.fit(fit, cont.matrix)

#Perform ebayes statistic
message('eBayes ...')
fit <- limma::eBayes(fit, trend = FALSE)

#Extract table of results, remove non-necessary heavy variables
message('results prep...')
tableDEA <- limma::topTable(
  fit,
  coef = 1,
  adjust.method = "fdr",
  number = nrow(TOC)
)
rm(fit)
rm(TOC)

#Get significant genes
tableDEA_sig<-tableDEA[tableDEA$adj.P.Val<0.05,]
tableDEA_sig_FC<-tableDEA_sig[abs(tableDEA_sig$logFC)>1,]

write.csv(tableDEA_sig_FC,file='compare_data/dataDEGs.csv')

#Save up-regulated significant genes
tableDEA_up<-tableDEA_sig_FC[tableDEA_sig_FC$logFC>0,]
write.csv(tableDEA_up,file='compare_data/dataDEGs_up.csv')

#Save down-regulated genes
tableDEA_down<-tableDEA_sig_FC[tableDEA_sig_FC$logFC<0,]
write.csv(tableDEA_down,file='compare_data/dataDEGs_down.csv')


