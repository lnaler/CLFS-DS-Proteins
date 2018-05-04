#!/usr/bin/env python
import os, quilt
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
import pandas_profiling

# Load Data
dir_name = os.path.abspath(os.path.dirname(__file__))
data_file = 'Data_Cortex_Nuclear.xls'
data_file_path = os.path.join(dir_name, data_file)
df = pd.read_excel(data_file_path,header=0)

names= list(df.columns.values)[1:-4]
prot_names=[x.encode('UTF8')[:-2] for x in names]

import missingno as msno
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
msno.matrix(df,inline=False)
ax0.set_xticklabels(np.repeat(prot_names, 2),
                    rotation=45, fontsize=8)

plt.savefig('missing_matrix.png')

