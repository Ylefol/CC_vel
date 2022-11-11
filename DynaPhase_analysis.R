#Dependencies

setwd("~/A_Projects/BiGR/BiGR_vel")

library("reshape2")
library("ggplot2")
library("gplots")
#library("gProfileR")
library("gprofiler2")
library("ggrepel")
library("stringr")
library("ReactomeContentService4R")

#Get members of REAC of interest
# REAC_1<-event2Ids(event.id = "R-HSA-2500257")$geneSymbol#Resolution of Sister Chromatid Cohesion
# REAC_2<-event2Ids(event.id = "R-HSA-68877")$geneSymbol #Mitotic Prometaphase
# REAC_3<-event2Ids(event.id = "R-HSA-72613")$geneSymbol#Eukaryotic Translation Initiation
# REAC_4<-event2Ids(event.id = "R-HSA-2408557")$geneSymbol#Selenocysteine synthesis
# write.table(REAC_1,'Resolution_of_Sister_Chromatid_Cohesion.txt',row.names = F,quote = F, col.names =F)
# write.table(REAC_2,'Mitotic_Prometaphase.txt',row.names = F,quote = F, col.names =F)
# write.table(REAC_3,'Eukaryotic_Translation_Initiation.txt',row.names = F,quote = F, col.names =F)
# write.table(REAC_4,'Selenocysteine_synthesis.txt',row.names = F,quote = F, col.names =F)



runProfiler<- function(Genelist, Group.No, max_p_value=0.05){
  GO.df <- gprofiler(Genelist, organism="hsapiens", significant=FALSE, max_p_value=max_p_value, correction_method="fdr", domain_size="annotated", src_filter = c("GO", "KEGG", "REAC"))
  if (nrow(GO.df) <= 0) return(NULL)
  Fisher.df <- data.frame(O.S = GO.df$overlap.size, Q.O = (GO.df$query.size - GO.df$overlap.size), Q.T = (GO.df$term.size - GO.df$overlap.size),
                          U = (25000 - GO.df$query.size - GO.df$term.size + GO.df$overlap.size))
  GO.df$OR <- apply(Fisher.df, 1, function(x) fisher.test(matrix(x, nr=2))$estimate)
  GO.df$Group <- rep(Group.No, times = length(GO.df$domain))
  return(GO.df)
}

runProfiler2<- function(Genelist, Group.No, max_p_value=0.05){
  GO.df <- gost(Genelist, organism="hsapiens", significant=TRUE, user_threshold=max_p_value, correction_method="fdr", domain_scope="annotated", sources = c("GO", "KEGG", "REAC"))
  GO.df <- GO.df$result
  if (nrow(GO.df) <= 0) return(NULL)
  Fisher.df <- data.frame(O.S = GO.df$intersection_size, Q.O = (GO.df$query_size - GO.df$intersection_size), Q.T = (GO.df$term_size - GO.df$intersection_size),
                          U = (25000 - GO.df$query_size - GO.df$term_size + GO.df$intersection_size))
  GO.df$OR <- apply(Fisher.df, 1, function(x) fisher.test(matrix(x, nr=2))$estimate)
  GO.df$Group <- Group.No
  return(GO.df)
}

cellcycle.rea <- getParticipants("R-HSA-1640170", retrieval = "EventsInPathways")
cellcycle.ids <- c("R-HSA-1640170", cellcycle.rea$stId)

HaCaT <- na.omit(read.table("data_files/data_results/rank/HaCat/A_B_t_test_results.csv", header=T, sep=","))

#spliced_up_CI_phase  => "Upper confidence interval is below 0 -> negative velocity"

GetTopTerms <- function(tab, top=10) {return(tab[order(tab$p_value, 1 / tab$OR), ][1:top,"term_name"])}

GetCCTerms <- function(tab) {return(tab[grep("(G0|G1[/ ]|G2|G0 and Early G1|(S|M|G1) Phase|[Mm]itotic .*[Pp]hase|[Cc]ell [Cc]ycle)", tab$term_name),"term_name"])}

GetCCReactome <- function(tab) {return(tab[sub("REAC:", "", tab$term_id) %in% cellcycle.ids,"term_name"])}

MeltProfiler1 <- function(list) {
  vars <-  c("p_value", "term_name", "OR")
  melt(lapply(list, function(x) x[, vars]), id.vars=vars)
}

MeltProfiler2 <- function(list) {
  vars <-  c("p_value", "term_name", "OR")
  melt(lapply(list, function(x) lapply(x, function(y) y[, vars])), id.vars=vars)
}

PlotTerms <- function(list, terms, melter=MeltProfiler1, phases=c("G1", "S", "G2M")) {
  pd <- melter(list)
  pd <- pd[pd$term_name %in% terms,]
  pd$logP <- -log10(pd$p_value)
  if ("L2" %in% colnames(pd)) {
     pd$tmp <- pd$L1
     pd$L1 <- as.character(pd$L2)
     pd$L2 <- as.character(pd$tmp)
     pd$L2 <- factor(pd$L2, levels=names(list), ordered=T)
  }
  pd$L1 <- factor(pd$L1, levels=phases, ordered=T)
  pd <- pd[order(pd$L1, pd$p_value),]
  
  pd$term_name <- str_wrap(pd$term_name, width=100)
  pd$term_name <- factor(pd$term_name, levels=rev(unique(pd$term_name)), ordered=TRUE)

  p <- ggplot(pd, aes(L1, term_name)) + geom_point(aes(color=logP, size=OR)) + theme_bw() + scale_colour_gradient(low=("blue"), high=("red")) +
    labs(x="Phase", y="Reactome pathway", color=expression(paste(-log[10], " FDR")), size="OR")
  if ("L2" %in% colnames(pd)) {
     p <- p + facet_grid(.~L2) 
  }
  return(p)
}

  
phases <- c("G1", "S", "G2M")

# HaCat peak phase
go.results.peaks <- sapply(unique(na.omit(HaCaT$phase_peak_exp)), function(x) runProfiler2(as.character(HaCaT$gene_name[HaCaT$phase_peak_exp %in% x]), x, 0.05), simplify=F)
go.results.peaks.rea <- lapply(go.results.peaks, function(x) {x<-x[x$source %in% "REAC",]; x[order(x$p_value),]})

cc.r <- unique(unlist(lapply(go.results.peaks.rea, GetCCReactome)))
cc.top <- unique(na.omit(unlist(lapply(lapply(go.results.peaks.rea[phases], function(x) x[x$term_name %in% cc.r, ]), GetTopTerms, 10))))
ggsave("HaCat_reactome_CC_peak_exp.pdf", PlotTerms(go.results.peaks.rea[phases], cc.top), width=7.5, height=5)


top <- unique(unlist(lapply(go.results.peaks.rea, GetTopTerms, 5)))
ggsave("HaCat_reactome_peak_exp.pdf", PlotTerms(go.results.peaks.rea[phases], top), width=7.5, height=7)


#HaCaT Dynamic phases
HaCaT$DynPhase <- paste(HaCaT$phase_start_vel, HaCaT$phase_peak_vel, sep="\n")
go.results.dynphase_HaCaT <- sapply(unique(na.omit(HaCaT$DynPhase)), function(x) runProfiler2(as.character(HaCaT$gene_name[HaCaT$DynPhase %in% x]), x, 0.05), simplify=F)
go.results.dynphase.rea_HaCat <- lapply(go.results.dynphase_HaCaT, function(x) {x<-x[x$source %in% "REAC",]; x[order(x$p_value),]})

combPhases_HaCat <- paste(sapply(phases, rep, 3), phases, sep="\n")

top_HaCat <- unique(unlist(lapply(go.results.dynphase.rea_HaCat, GetTopTerms)))
ggsave("HaCat-DynPhases_reactome.pdf", PlotTerms(go.results.dynphase.rea_HaCat, top_HaCat, phases=combPhases_HaCat), width=9, height=10)

cc.r_HaCat <- unique(unlist(lapply(go.results.dynphase.rea_HaCat, GetCCReactome)))
cc.top_HaCat <- unique(na.omit(unlist(lapply(lapply(go.results.dynphase.rea_HaCat, function(x) x[x$term_name %in% cc.r_HaCat, ]), GetTopTerms, 10))))
ggsave("HaCat-DynPhases_reactome_CC.pdf", PlotTerms(go.results.dynphase.rea_HaCat, cc.top_HaCat, phases=combPhases_HaCat), width=9, height=7)



#Compare cell lines with peak expression
c293T <- na.omit(read.table("data_files/data_results/rank/293t/A_B_C_D_t_test_results.csv", header=T, sep=","))
Jurkat <- na.omit(read.table("data_files/data_results/rank/jurkat/A_B_C_D_t_test_results.csv", header=T, sep=","))
go.results.comb <-
  lapply(list(HaCaT=HaCaT, "293T"=c293T, Jurkat=Jurkat),
         function(cell) sapply(unique(na.omit(cell$phase_peak_exp)), function(x) runProfiler2(as.character(cell$gene_name[cell$phase_peak_exp %in% x]), x, 0.05), simplify=F))
go.results.comb.rea <- lapply(go.results.comb, function(res) lapply(res, function(x) {x<-x[x$source %in% "REAC",]; x[order(x$p_value),]}))
go.results.comb.bp <- lapply(go.results.comb, function(res) lapply(res, function(x) {x<-x[x$source %in% "GO:BP",]; x[order(x$p_value),]}))

cc.r2 <- unique(unlist(lapply(go.results.comb.rea, function(x) lapply(x, GetCCReactome))))
cc.top2 <- unique(na.omit(unlist(lapply(go.results.comb.rea, function(list) lapply(lapply(list[phases], function(x) x[x$term_name %in% cc.r2, ]), GetTopTerms, 10)))))
ggsave("All_reactome_CC_peak_exp.pdf", PlotTerms(go.results.comb.rea, na.omit(cc.top2), MeltProfiler2), width=10, height=6)

top2 <- unique(unlist(lapply(go.results.comb.rea, function(x) lapply(x, GetTopTerms, 5))))
ggsave("All_reactome_peak_exp.pdf", PlotTerms(go.results.comb.rea, na.omit(top2), MeltProfiler2), width=10, height=6)