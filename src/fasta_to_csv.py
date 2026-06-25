import os

def convert_fasta_to_csv(fasta_path, csv_path):
    """
    Converts a FASTA file to CSV format, with columns protein_name, sequence and length.
    """

    try:
        with open(fasta_path, "r", encoding="utf-8") as f_fasta, open(csv_path, "w", encoding="utf-8") as f_csv:
            # Name the columns of the .csv file
            f_csv.write("protein_name,sequence,length\n")
            
            for line in f_fasta:
                if line.startswith(">"):
                    f_csv.write(f"{line[1:-1]},")
                else:
                    length=len(line)-1
                    f_csv.write(f"{line[:-1]},{length}\n")
        return None
    except FileNotFoundError:
        print(f"File not found.")
        return None


if __name__ == "__main__":
    # Paths are defined relative to the current script location (src/)
    
    # Takes all FASTA files to convert them to CSV format
    for name in os.listdir("../data/fasta_files/training"):
        if name.endswith(".fasta"):
            alt_name = name.replace(".fasta", ".csv")
            input_path = f"../data/fasta_files/training/{name}"
            output_path = f"../data/csv_files/training/{alt_name}"
            # Creates the CSV file
            convert_fasta_to_csv(input_path, output_path)