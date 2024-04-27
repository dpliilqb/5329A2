import numpy as np
import re
import pandas as pd
from io import StringIO

def load_data(path):
    with open(path) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # print(df)
    return df

if __name__ == "__main__":
    FILENAME = 'train.csv'
    df = load_data(FILENAME)