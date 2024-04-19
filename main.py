import numpy as np
import re
import pandas as pd
from io import StringIO




if __name__ == "__main__":
    FILENAME = 'train.csv'
    with open(FILENAME) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    print(df)