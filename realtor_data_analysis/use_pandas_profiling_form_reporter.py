import pandas_profiling
import pandas as pd

# 这个对内存的要求很大，如果机器不好可能无法运行
data = pd.read_csv('./input/total_realor_data.csv')
prf = pandas_profiling.ProfileReport(data)
prf.to_file('./realtor_data_reporter.html')