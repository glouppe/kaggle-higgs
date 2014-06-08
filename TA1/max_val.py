import pandas as pd
import numpy as np


p1 = pd.read_csv('gil1.csv')
p2 = pd.read_csv('gil2.csv')
p3 = pd.read_csv('tris1.csv')
p4 = pd.read_csv('tris2.csv')
p5 = pd.read_csv('combined_new_8_.csv')
p6 = pd.read_csv('combined_new_9.csv')
p7 = pd.read_csv('combined_new_10.csv')
p8 = pd.read_csv('combined_new_11.csv')
p9 = pd.read_csv('combined_new_12.csv')
p10 = pd.read_csv('combined_new_13.csv')
p11 = pd.read_csv('combined_new_14.csv')
p12 = pd.read_csv('combined_new_14.csv')
p13 = pd.read_csv('combined_new_15.csv')
p14 = pd.read_csv('combined_new_16.csv')


all_preds = np.column_stack((p1['Class'].values,p2['Class'].values,p3['Class'].values,p4['Class'].values,p5['Class'].values,p6['Class'].values, 
							p7['Class'].values, p8['Class'].values,p9['Class'].values,p10['Class'].values,p11['Class'].values,p12['Class'].values,
							p13['Class'].values,p14['Class'].values

	))

preds = []

for i in range(all_preds.shape[0]):
	if list(all_preds[i,:]).count('s') >=  list(all_preds[i,:]).count('b'):
		preds.append('s')
	else:
		preds.append('b')

p3['Class'] = preds

p3.to_csv('combined_Gil_1.csv', index = False)

