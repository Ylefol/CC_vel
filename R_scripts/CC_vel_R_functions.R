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
    labs(x="Phase", y="Reactome pathway", color=expression(paste(-log[10], " FDR")), size="OR") +
    theme(text = element_text(size = 15))
  if ("L2" %in% colnames(pd)) {
    p <- p + facet_grid(.~L2) 
  }
  p <-p + scale_x_discrete(labels=c("G1"='G1',"S"='S','G2M'="G2/M"))
  return(p)
}