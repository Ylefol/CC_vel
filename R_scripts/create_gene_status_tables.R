


target_cell_lines<-c('HaCat-A_B','jurkat-A_B_C_D','293t-A_B_C_D')
library("xlsx")
file_create<-F

for(cell_line in target_cell_lines){
  cell_name<-strsplit(cell_line,'-')[[1]][1]
  replicate<-strsplit(cell_line,'-')[[1]][2]
  
  t_result<-read.csv(paste0('A_Projects/BiGR/CC_vel/data_files/data_results/rank/',cell_name,'/',replicate,'_t_test_results.csv'))
  var_results<-read.csv(paste0('A_Projects/BiGR/CC_vel/',cell_name,'_var.csv'))
  colnames(var_results)<-c('gene_name','log10_variability')
 
  merged_dta<-merge(t_result,var_results,by='gene_name')
 
  merged_dta$status<-rep('None',nrow(merged_dta))
  merged_dta$status[merged_dta$t>0]<-'Rank-able'
  merged_dta$status[merged_dta$t>0 & merged_dta$padjusted<0.01]<-'Significant padj'
  merged_dta$status[merged_dta$t>0 & merged_dta$padjusted<0.01 & merged_dta$log10_variability>0.001]<-'Significant padj and var'
 
  #Reorganize columns
  my_cols<-colnames(merged_dta)
  my_cols<-my_cols[c(1,9,2,3,4,8,5,6,7)]
  merged_dta<-merged_dta[,my_cols]
  
  # Write the first data set in a new workbook
  write.xlsx(merged_dta, file = "CC_vel_gene_status.xlsx",
             sheetName = cell_name, append = file_create, row.names=F)
  if(file_create==F){
    file_create=T
  }
}

