# BiGR_vel
BiGR_vel takes in single cell RNAseq data, establishes a cell cycle based pseudotime and calculates the cell cycle time derivative for each gene.


## Overview
BiGR_vel was developed within the BiGR group at NTNU. It takes in loom files generate by the velocyto command line using a list of snakemake rules found [here](https://github.com/gcfntnu/single-cell/blob/master/rules/quant/velocyto.rules).
BiGR_vel then creates a cell cycle based pseudotime using the expression of know cell cycle genes. This is followed up by a RNA velocity analysis using velocyto and finalized by the calculation of gene time derivative throughout the established cell cycle pseudotime

## Merging of replicates
RNA velocity and gene velocity have the consequence of being unreliable with current data. To accommodate for this, BiGR_vel allows the use of multiple replicates and multiple iterations. During the development of this tool, we used two technical replicates of a HaCat cell line. We established a cell cycle pseudotime for each technical replicate and performed RNA velocity on each in order to verify their similarity. We then proceeded to calculate gene velocity and merge these values across replicates and iterations.

The calculation of gene velocity depends on RNA velocity, which it depends on some random variables, we therefore perform several (5) iterations for each replicate and then proceed to merge the now 10 runs. This merged data now represents gene velocity along with a confidence interval allowing for better certainty of the observed values.

## Installation
BiGR_vel does not come in a library or package format, it is instead a set of snakemake rules. To use this tool it is advised to create a mniniconda environment and install the packages listed in the 'Dependencies' section fo this ReadMe. With this done, this repository can be cloned and each snakemake rule can be run one at a time. A detailed workflow of which rules to run, how to run them, and their requirements are listed below in the 'Tutorial' section.


## Dependencies
Main dependencies are:
Python 3, pandas, numpy, scipy.stats
Essential packages are: 
velocyto, scanpy, pyranges, logging, pickle
Plotting packages are:
ptitprince, seaborn, matplotlib, rpy2

An exhaustive list of all packages contained in the conda environment can be found in the 'miniconda_env_packages.txt' file within this repository.


## Tutorial - Snakemake
There are five rules to run the complete pipeline. The duration of the pipeline will vary greatly based on the parameters used within each rule.

### Rule #1
The first rule takes in loom files and performs some quality control filtering and cell cycle phase reassignement. One of the quality controls is to set a threshold based on identified boundaries. As this has to be done manually, the rule will request user input. For this reason, it is very important to run the rule on a single core, otherwise pop-up images will not be distinguishable from one run to another.
Note that this rule is quick, therefore the time investment here is minimal.

This rule also removes cells indicated by the barcode text file. We utilized this since our dataset contained both HaCat and murine cells, however the murine cells were not used, therefore we removed them. This file should be adapted where barcodes to remove should be contained within it, if no barcodes should be removed, leave the file empty.
Each sample/replicate should have it's own representative barcode file.
Lastly, to identify mitochondrial reads, the rule requires mitochondrial gene names, these are made available via the chrM_unique.txt file, though if the organism used is not human, this list may have to be replaced with another. The same applies to the cell cycle genes which are specific to human cells.

Example run:
`snakemake data_files/phase_reassigned/CC_HaCat_{A_B}.loom --cores 1`
This will run both replicates of the HaCat cell line on a single core, indicating that they will be run one at a time.
These runs will create two plots, from which three thresholds will have to be inputted.
The first is the percentage of mitochondrial reads, from which an upper threshold will have to be set. This threshold will be the 'top' of the bulb of the violin plot.
The two next thresholds will be based on the violin plot representing unspliced reads. The lower and upper thresholds should be the percentage representing the lower and upper area of the bulb within the violin respectively.
NOTE: Thresholds should be in percents, a value of 0.25 should be inputed as 25


### Rule #2
The next rule is much simpler. This rule performs the RNA velocity to perform a validation of the cell cycle pseudotime and also serves as a means to identify the number of neighbours to use to identify the best RNA velocity patterns.
RNA velocity looks for the directionality of each cell based on the nearest neighbour approach, however we can specify the number of neighbours to look at and this is dependant on the number of cells in each dataset. With a dataset of ~1500 cells (post filer/first rule) we found that a num_k of 550 worked well, so roughly 1/3 of the dataset.

`snakemake data_files/boundary_data/HaCat_{A,B}_boundaries.csv --cores 2`
This rule is run on two cores to perform both A and B at the same time.


### Rule #3
This rule performs the number of necessary iterations for each replicate that will be merged downstream (in the next rule). Here we have to specify the same k as in Rule #2 as well as the number of desired iterations. We found 5 to be adequate for clean datasets, though higher numbers may be beneficial for other datasets.

`snakemake data_files/confidence_intervals/HaCat/{A,B}/Iterations --cores 2`

Note that this rule can get RAM intensive depending on the datasets and therefore the numbers of cores should be used with caution.

### Rule #4
This rule merges the various replicates of the dataset, in our case, A and B.
Along with the merger, it calculates the confidence interval, which can be customized based on the set z value.

To merge A and B, the rule is run as follows.
`snakemake data_files/confidence_intervals/HaCat/merged_results/A_B --cores 1`
If one were to calculate gene velocity without merging replicates, the rule can be run using only A or B (or whatever other replicate naming convention)


### Rule #5
This rule calculates several results that may be used in custom analyses. It starts by calculating the delay of each gene (distance between unspliced and spliced at a cross-over point) as well as performing a ranking using a t-test. Along with the t-test, it determines where genes peak in expression along the cell cycle.

`snakemake data_files/data_results/rank/HaCat/A_B_ranked_genes.csv --cores 1`

### Rule #6
Rule #6 is not part of the standard pipeline, but is instead part of a microRNA analysis. This rule takes in smallRNAseq data along with targetscan data. It then creates csv files of microRNAs which fit within a specified threshold ex: 0_100 or 1000_None). These files contain the microRNAs which fit within the threshold and that have been categorized.


## Tutorial - analysis
Using the results from the above rules, we can create non-default results using either the analysis script or the miRNA analysis script (if miRNA results are desired).
The analysis scripts show how to retrieve the data that was generated and show some basic plots that have already been designed.
These scripts were meant as a template for users, where they can familiarize themselves with the plots and then perform their own analyses with the generated results.



## Tutorial DynaPhase_analysis
An R script is provided to generate the dotplots shown in our article. These dotplots served as a means to perform a gprofiler REACTOME based over-representation analysis for the genes which peaked in each cell cycle phase separately. We used this to validate our cell cycle pseudotime.





