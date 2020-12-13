import csv

def load_config(config_csv):
    configs = []
    with open(config_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for config in reader:
            configs.append(config)
    return configs