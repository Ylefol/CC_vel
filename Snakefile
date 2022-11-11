#-*- mode:snakemake -*-


#Rule should only be run on one core due to inputs required.
#The rule will generate two plots which must be interpreted by the user
#Who will be prompted to input three numbers, one after the other, to complete
#the preprocessing
#Barcode can be set to None if no cell barcodes need to be filtered out
rule basic_filter_and_reassignment:
    input:
        "data_files/initial_data/loom_files/{cell_line}_{replicate}.loom"
    params:
        script="snake_scripts/snake_reassignment.py",
        barcodes="data_files/initial_data/barcodes/{cell_line}_{replicate}_barcodes.txt", #Can be None
        mito_genes="data_files/initial_data/chrM_unique.txt",
        CC_genes="data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv"
    output:
        cc_loom="data_files/phase_reassigned/CC_{cell_line}_{replicate}.loom",
        all_gene_loom="data_files/phase_reassigned/AG_{cell_line}_{replicate}.loom"
    log:
        "logs/phase_reassignment_logs/{cell_line}_{replicate}.log"
    shell:
        "python {params.script} 2> {log} {input} {params.barcodes} {params.mito_genes} {params.CC_genes} {output.cc_loom} {output.all_gene_loom}"


#This rule runs velocyto on cell cycle genes only
#Take care to adjust the num_k (number of neighbours) for your dataset
rule velocyto_CC:
    input:
        "data_files/phase_reassigned/CC_{cell_line}_{replicate}.loom"
    params:
        script='snake_scripts/snake_velocyto.py',
        num_k=220
    output:
        boundaries_csv="data_files/boundary_data/{cell_line}_{replicate}_boundaries.csv"
    log:
        "logs/velocyto_logs/{cell_line}_{replicate}.log"
    shell:
        "python {params.script} 2> {log} {input} {output.boundaries_csv} {params.num_k}"


#Runs n iterations of velocyto on all genes in order to apply the downsampling
#and other filter n times. The n iterations are then merged
#This rule is quite heavy on RAM and therefore the number of cores should be selected
#with RAM limitations in mind.
#Using a computer with 32Gb of RAM, running more than 2-3 replicates at once is
#the maximum
#Iterations are performed for velocity and expression values
rule velocyto_iterations:
    input:
        "data_files/phase_reassigned/AG_{cell_line}_{replicate}.loom"
    params:
        script='snake_scripts/snake_vel_iterations.py',
        number_of_iterations=5,
        num_k=220
    output:
        vel_Iterations=directory('data_files/confidence_intervals/{cell_line}/{replicate}/Iterations')
    log:
        "logs/velocyto_CI_logs/{cell_line}_{replicate}.log"
    shell:
        "python {params.script} 2> {log} {input} {params.number_of_iterations} {params.num_k}"

#Calculate the confidence intervals for individual or merged replicates
#90 - 1.645     95 - 1.96       99-2.576
#This can be calculated on single replicates or on merged replicates, if merging replicates
#the replicates need to be split with a '_'. For example, if we have two replicates
#and we want to run the rule for both replicates individually as well as merged, we would
#input the following {A,B,A_B} in the 'replicate' area.
rule compute_merged_CIs:
    input:
        'data_files/confidence_intervals/{cell_line}'
    params:
        script='snake_scripts/snake_merge_reps_calculate_CIs.py',
        number_of_iterations=5,
        z_CI_value=2.576
    output:
        merged_res=directory('data_files/confidence_intervals/{cell_line}/merged_results/{replicate}')
    log:
        "logs/merged_CI_logs/{cell_line}_{replicate}.log"
    shell:
        "python {params.script} 2> {log} {input} {params.number_of_iterations} {params.z_CI_value} {output.merged_res}"
        
        
#Ranks the genes using a t test, and finds the delay for each gene.
rule t_test_delay:
    input:
        'data_files/confidence_intervals/{cell_line}/merged_results/{replicate}'
    params:
        script='snake_scripts/snake_gene_rank_delay.py',
        num_iters=5 
    output:
        ranked_genes='data_files/data_results/rank/{cell_line}/{replicate}_ranked_genes.csv',
        delay_genes='data_files/data_results/delay_genes/{cell_line}/{replicate}_delay_genes.csv',
        t_test_res='data_files/data_results/delay_genes/{cell_line}/{replicate}_t_test_results.csv'
    log:
        'logs/t_test_delay_logs/{cell_line}_{replicate}.log'
    shell:
        "python {params.script} 2> {log} {input} {params.num_iters} {output.ranked_genes} {output.delay_genes} {output.t_test_res}"



#Min max threshold must be split by a '_', can include none for the max
#Takes in smallRNA sequencing count files and prepares them to be used in miRNA analysis
rule miRNA_prep:
    input:
        "data_files/miRNA_files/non_categorized/{cell_line}_miRNA.csv"
    params:
        script="snake_scripts/snake_miRNA_prep.py"
    output:
        "data_files/miRNA_files/categorized/{cell_line}_miRNA_{min_max_threshold}.csv"
    log:
        "logs/miRNA_logs/{cell_line}_{min_max_threshold}.log"
    shell:
        "python {params.script} 2> {log} {input} {output}"
        