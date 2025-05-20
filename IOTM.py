import pandas as pd

# Load your dataset
df = pd.read_csv("merged_cleaned_labeled_dataset.csv")  # replace with your actual file name

# Map string labels to numeric
label_map = {
    'Benign': 0,
    'BruteForce': 1,
    'DNSSpoofing': 2,
    'ARPSpoofing': 3
}
df['Class'] = df['Label'].map(label_map)

# Select only useful columns (you can tweak this list if needed)
columns_to_keep = [
    'Protocol', 'Flow Duration',
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    'Label', 'Class'
]

# Filter dataframe
df_filtered = df[columns_to_keep]

# Save cleaned file
df_filtered.to_csv("cleaned_dataset.csv", index=False)

print("Clean dataset saved as cleaned_dataset.csv")

"""import pandas as pd

# Define file-label mappings
file_label_map = {
    "BenignTraffic.pcap_Flow.csv": "Benign",
    "DictionaryBruteForce.pcap_Flow.csv": "BruteForce",
    "DNS_Spoofing.pcap_Flow.csv": "DNSSpoofing",
    "MITM-ArpSpoofing.pcap_Flow.csv": "ARPSpoofing"
}

# Columns to remove (non-informative or ID-based)
cols_to_drop = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
    'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
    'Label'  # Remove original label to overwrite it
]

# Initialize empty list to store dataframes
merged_dataframes = []

# Load, clean, and label each dataset
for filename, label in file_label_map.items():
    df = pd.read_csv(filename)

    # Drop unnecessary columns if present
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Assign new label
    df['Label'] = label

    merged_dataframes.append(df)

# Merge all data
final_dataset = pd.concat(merged_dataframes, ignore_index=True)

# Save to CSV
final_dataset.to_csv("merged_cleaned_labeled_dataset.csv", index=False)

print("Dataset cleaned, labeled, and saved as 'merged_cleaned_labeled_dataset.csv'")"""
