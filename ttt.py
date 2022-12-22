import pandas as pd
import os
data_dir = './data'
public_x_csv = os.path.join(data_dir, 'public_x_alert_date.csv')
public_private_x_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')
df_public_x = pd.read_csv(public_x_csv)
df_public_private_x = pd.read_csv(public_private_x_csv)
print(df_public_x[0])
# print(df_public_x['alert_key'])
# print(df_public_private_x['alert_key'])
