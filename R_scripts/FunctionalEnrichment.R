

setwd("~/A_Projects/BiGR/CC_vel")

library("reshape2")
library("ggplot2")
library("gplots")
library("gprofiler2")
library("ggrepel")
library("stringr")
library("ReactomeContentService4R")

source('R_scripts/CC_vel_R_functions.R')


#Get results in t-test format
HaCaT <- na.omit(read.table("data_files/data_results/rank/HaCat/A_B_t_test_results.csv", header=T, sep=","))
# Open file that contains genes of interest (t-test based, variability filter, and delay threshold)
HaCaT_genes_interest<-read.table('data_files/data_results/HaCat_all_sig.txt')
#Subset
HaCaT<-HaCaT[HaCaT$gene_name %in% HaCaT_genes_interest$V1,]

  
phases <- c("G1", "S", "G2M")

# HaCat peak phase
go.results.peaks <- sapply(unique(na.omit(HaCaT$phase_peak_exp)), function(x) runProfiler2(as.character(HaCaT$gene_name[HaCaT$phase_peak_exp %in% x]), x, 0.05), simplify=F)
go.results.peaks.rea <- lapply(go.results.peaks, function(x) {x<-x[x$source %in% "REAC",]; x[order(x$p_value),]})

#Get top 10 and plot + save
top <- unique(unlist(lapply(go.results.peaks.rea, GetTopTerms, 10)))
ggsave("HaCat_reactome_peak_exp.pdf", PlotTerms(go.results.peaks.rea[phases], top), width=10, height=9)



#Get 293t results
c293T <- na.omit(read.table("data_files/data_results/rank/293t/A_B_C_D_t_test_results.csv", header=T, sep=","))
c293t_genes_interest<-read.table('data_files/data_results/293t_all_sig.txt')
c293T<-c293T[c293T$gene_name %in% c293t_genes_interest$V1,]

#Get Jurkat results
Jurkat <- na.omit(read.table("data_files/data_results/rank/jurkat/A_B_C_D_t_test_results.csv", header=T, sep=","))
jurkat_genes_interest<-read.table('data_files/data_results/jurkat_all_sig.txt')
Jurkat<-Jurkat[Jurkat$gene_name %in% jurkat_genes_interest$V1,]

#Get the results from phase_peak_exp for each cell line
go.results.comb <-
  lapply(list(HaCaT=HaCaT, "293T"=c293T, Jurkat=Jurkat),
         function(cell) sapply(unique(na.omit(cell$phase_peak_exp)), function(x) runProfiler2(as.character(cell$gene_name[cell$phase_peak_exp %in% x]), x, 0.05), simplify=F))
go.results.comb.rea <- lapply(go.results.comb, function(res) lapply(res, function(x) {x<-x[x$source %in% "REAC",]; x[order(x$p_value),]}))
go.results.comb.bp <- lapply(go.results.comb, function(res) lapply(res, function(x) {x<-x[x$source %in% "GO:BP",]; x[order(x$p_value),]}))

#Plot the results for all three cell lines using top 5
top2 <- unique(unlist(lapply(go.results.comb.rea, function(x) lapply(x, GetTopTerms, 5))))
ggsave("All_reactome_peak_exp_5.pdf", PlotTerms(go.results.comb.rea, na.omit(top2), MeltProfiler2), width=12, height=12)
