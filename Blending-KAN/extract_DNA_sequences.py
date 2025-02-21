import argparse
import pandas as pd
from pyfaidx import Fasta
## pip install pandas
## pip install pyfaidx

'''
demo

python extract_DNA_sequences.py \
    --input_csv predictions_with_probabilities.csv \
    --genome_fasta Homo_sapiens.GRCh38.dna.toplevel.fa \
    --output_bed positive_samples1.bed \
    --output_fasta positive_samples1.fasta  \
    --output_csv output_sequences11.csv
    
Notes:


head predictions_with_probabilities.csv
Sample_ID,Predicted_Class,Probability_Class_0,Probability_Class_1,Chromosome,Start_Position,End_Position
0,1,0.40294233,0.5970577,chr1,1005800,1007301
1,1,0.12454599,0.875454,chr1,1005960,1007461
2,1,0.25905165,0.7409483,chr1,1304100,1305601
3,1,0.40294233,0.5970577,chr1,1304304,1305805
4,1,0.40294233,0.5970577,chr1,1685591,1687092
5,0,0.96190983,0.03809011,chr1,2014407,2015908
6,0,0.97383547,0.02616448,chr1,2097340,2098841
7,1,0.12684299,0.873157,chr1,2139500,2141001
8,1,0.12684299,0.873157,chr1,3312540,3314041


head Homo_sapiens.GRCh38.dna.toplevel.fa
>1 dna:chromosome chromosome:GRCh38:1:1:248956422:1 REF
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN





	
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Extract positive class samples from prediction results, save as BED file, and extract corresponding sequences.")
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True,
        help="Path to the prediction results CSV file (e.g., 'predictions_with_probabilities.csv')"
    )
    parser.add_argument(
        '--output_bed', 
        type=str, 
        required=True,
        help="Path to save the output BED file containing positive class samples."
    )
    parser.add_argument(
        '--output_fasta', 
        type=str, 
        required=True,
        help="Path to save the output FASTA file containing sequences of positive class samples."
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True,
        help="Path to save the output CSV file containing sequences of positive class samples."
    )
    parser.add_argument(
        '--genome_fasta', 
        type=str, 
        required=True,
        help="Path to the genome reference FASTA file (e.g., 'Homo_sapiens.GRCh38.dna.toplevel.fa')"
    )
    return parser.parse_args()

def create_chromosome_mapping(genome_fasta):
    """
    创建染色体名称映射表，将 CSV 文件中的染色体名称（如 chr1）映射到 FASTA 文件中的染色体名称（如 1）。
    """
    genome = Fasta(genome_fasta)
    fasta_chromosomes = list(genome.keys())

    print("FASTA Chromosomes:", fasta_chromosomes)  # 打印FASTA文件中的染色体名称

    # CSV 文件中可能使用的染色体名称格式
    csv_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"] ## 如果是线粒体，或其他物种？？？？？ + ["A","B",....]

    csv_to_fasta_mapping = {}
    for csv_chrom in csv_chromosomes:
        csv_chrom_num = csv_chrom.replace("chr", "")  # 提取数字部分，如 chr1 -> 1
        if csv_chrom_num in fasta_chromosomes:
            csv_to_fasta_mapping[csv_chrom] = csv_chrom_num
        else:
            raise ValueError(f"Chromosome {csv_chrom} not found in the genome FASTA file.")

    print("Generated Chromosome Mapping:", csv_to_fasta_mapping)
    return csv_to_fasta_mapping

def extract_positive_samples(input_csv, output_bed, genome_fasta, output_fasta, output_csv):
    # Load the prediction results
    df = pd.read_csv(input_csv)
    
    # Filter rows where Predicted_Class == 1 (positive class)
    positive_df = df[df['Predicted_Class'] == 1]
    
    # Ensure the required columns are present
    required_columns = ['Chromosome', 'Start_Position', 'End_Position']
    if not all(col in positive_df.columns for col in required_columns):
        raise ValueError("The input CSV file is missing required columns. Ensure it contains 'Chromosome', 'Start_Position', and 'End_Position'.")
    
    # Extract relevant columns and rename them to match BED format
    bed_df = positive_df[['Chromosome', 'Start_Position', 'End_Position']]
    bed_df.columns = ['chrom', 'start', 'end']
    
    # Save as BED file
    bed_df.to_csv(output_bed, sep='\t', index=False, header=False)
    print(f"Positive class samples have been saved to {output_bed}")
    
    # Create chromosome mapping
    csv_to_fasta_mapping = create_chromosome_mapping(genome_fasta)
    
    # Extract sequences from the genome FASTA file
    genome = Fasta(genome_fasta)
    sequences = []  # 用于存储提取的序列
    for idx, row in bed_df.iterrows():
        csv_chrom = row['chrom']
        start = row['start']
        end = row['end']
        
        # Map CSV chromosome name to FASTA chromosome name
        fasta_chrom = csv_to_fasta_mapping.get(csv_chrom)
        if not fasta_chrom:
            raise KeyError(f"Chromosome {csv_chrom} not found in the mapping table.")
        
        if fasta_chrom not in genome:
            raise KeyError(f"Chromosome {fasta_chrom} not found in the genome FASTA file.")
        
        sequence = genome[fasta_chrom][start:end].seq
        sequences.append(sequence)  # 将序列添加到列表中
    
    # 保存为FASTA文件
    with open(output_fasta, 'w') as fasta_out:
        for idx, seq in enumerate(sequences):
            fasta_out.write(f">sequence_{idx + 1}\n{seq}\n")
    
    print(f"Sequences for positive class samples have been saved to {output_fasta}")
    
    # 保存为CSV文件
    sequence_df = pd.DataFrame(sequences, columns=['sequence'])
    sequence_df.to_csv(output_csv, index=False)
    print(f"Sequences for positive class samples have been saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    extract_positive_samples(args.input_csv, args.output_bed, args.genome_fasta, args.output_fasta, args.output_csv)