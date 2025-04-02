
library("ReactomeContentService4R")

#Get members of REAC of interest
REAC_1<-event2Ids(event.id = "R-HSA-2500257")$geneSymbol#Resolution of Sister Chromatid Cohesion
REAC_2<-event2Ids(event.id = "R-HSA-68877")$geneSymbol #Mitotic Prometaphase
REAC_3<-event2Ids(event.id = "R-HSA-72613")$geneSymbol#Eukaryotic Translation Initiation
REAC_4<-event2Ids(event.id = "R-HSA-2408557")$geneSymbol#Selenocysteine synthesis
write.table(REAC_1,'Resolution_of_Sister_Chromatid_Cohesion.txt',row.names = F,quote = F, col.names =F)
write.table(REAC_2,'Mitotic_Prometaphase.txt',row.names = F,quote = F, col.names =F)
write.table(REAC_3,'Eukaryotic_Translation_Initiation.txt',row.names = F,quote = F, col.names =F)
write.table(REAC_4,'Selenocysteine_synthesis.txt',row.names = F,quote = F, col.names =F)