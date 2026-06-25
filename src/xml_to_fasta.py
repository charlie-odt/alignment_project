import os
from lxml import etree # Used to parse XML and bypass the DOCTYPE issue

def convert_macsim_file(xml_path, fasta_path, txt_path):
    """
    Converts a MACSIM XML file to FASTA format and extracts annotations to TXT.
    """
    try:
        # Create a parser that ignores external DTDs to avoid network/DOCTYPE errors
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False)
        tree = etree.parse(xml_path, parser=parser)
        root = tree.getroot()

        with open(fasta_path, "w", encoding="utf-8") as f_fasta, \
             open(txt_path, "w", encoding="utf-8") as f_txt:
             
            count = 0
            for seq in root.findall(".//sequence"):
                seq_name_node = seq.find("seq-name")
                seq_data_node = seq.find(".//seq-data")

                if seq_name_node is not None and seq_data_node is not None:
                    name = seq_name_node.text.strip()
                    # Clean spaces and line breaks (keeps alignment dashes '-')
                    sequence = "".join(seq_data_node.text.split())

                    # 1. Write data to FASTA file
                    f_fasta.write(f">{name}\n{sequence}\n")
                    count += 1
                    
                    # 2. Add sequence to TXT file
                    # Calculate actual length (without gaps '-') for the end position
                    real_length = len(sequence.replace("-", ""))
                    
                    # Format: Description \t Seq_ID \t -1 \t Start \t End \t Feature_Type
                    seq_line = f"{sequence}\t{name}\t-1\t1\t{real_length}\tRaw_Sequence\n"
                    f_txt.write(seq_line)

                    # 3. Retrieve other annotations if any (Optional)
                    for feature in seq.findall(".//feature"):
                        f_type = feature.find("type")
                        f_name = feature.find("name")
                        f_start = feature.find("start")
                        f_end = feature.find("end")
                        
                        val_type = f_type.text.strip() if f_type is not None else "Annotation"
                        val_name = f_name.text.strip() if f_name is not None else val_type
                        val_start = f_start.text.strip() if f_start is not None else "0"
                        val_end = f_end.text.strip() if f_end is not None else "0"
                        
                        feature_line = f"{val_name}\t{name}\t-1\t{val_start}\t{val_end}\t{val_type}\n"
                        f_txt.write(feature_line)

        return count
        
    except Exception as e:
        print(f"Error reading {xml_path}: {e}")
        return 0


def process_directory(source_dir, output_dir):
    """
    Scans the source directory and converts all XML files.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return

    files = os.listdir(source_dir)
    xml_files = [f for f in files if f.lower().endswith(".xml")]

    if not xml_files:
        print(f"No .xml files found in directory: {source_dir}")
        return

    print(f"{len(xml_files)} XML files detected\n")

    total_sequences = 0
    converted_files = 0

    for idx, file_name in enumerate(xml_files, 1):
        xml_path = os.path.join(source_dir, file_name)

        fasta_name = os.path.splitext(file_name)[0] + ".fasta"
        txt_name = os.path.splitext(file_name)[0] + ".txt"

        fasta_path = os.path.join(output_dir, fasta_name)
        txt_path = os.path.join(output_dir, txt_name)

        nb_seq = convert_macsim_file(xml_path, fasta_path, txt_path)

        if nb_seq > 0:
            print(f"[{idx}/{len(xml_files)}] {file_name} -> {fasta_name} & {txt_name} ({nb_seq} sequences)")
            total_sequences += nb_seq
            converted_files += 1

    print(f"\n{converted_files} file(s) successfully converted (FASTA + TXT).")


if __name__ == "__main__":
    # Paths are defined relative to the current script location (src/)
    INPUT_DIR = os.path.join("..", "training")
    OUTPUT_DIR = os.path.join("..", "data", "fasta_files", "training")

    process_directory(INPUT_DIR, OUTPUT_DIR)