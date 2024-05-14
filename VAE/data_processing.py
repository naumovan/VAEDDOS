import torch
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from VAE.VAEarchitecture import VAE
from VAE.dataset import TSDataset

test_ratio = 0.001
quant = 0.999
output_file_path = '../outputs_after_vae/datatest_prepared.csv'
datatestfile_path = Path('/Users/user/Projects/VAEDDOS/data/train/1st nightframe.csv')
datasets_root = Path('/Users/user/Projects/VAEDDOS/VAE')

test_data = pd.read_csv(datatestfile_path)
test_data['Stable'] = "True"
test_data['Processing_time'] = np.random.randint(1, 4, size=len(test_data))
test_data['Source'] = test_data['Source'].str.replace('.', '')
test_data['Destination'] = test_data['Destination'].str.replace('.', '')
test_data.drop(columns=['No.','Info'])

print(test_data.head())
cont_vars = ['Length','Processing_time','Source','Destination']
cat_vars = ['Stable','Protocol']

label_encoders = [LabelEncoder() for _ in cat_vars]
for col, enc in zip(cat_vars, label_encoders):
    test_data[col] = enc.fit_transform(test_data[col])

tr_data = test_data.iloc[: int(len(test_data) * (1 - test_ratio))]
tst_data = test_data.iloc[int(len(test_data) * (1 - test_ratio)) :]
scaler = preprocessing.StandardScaler().fit(tr_data[cont_vars])

tr_data_scaled = tr_data.copy()
tr_data_scaled[cont_vars] = scaler.transform(tr_data[cont_vars])
tst_data_scaled = tst_data.copy()
tst_data_scaled[cont_vars] = scaler.transform(tst_data[cont_vars])

tr_data_scaled.to_csv('train.csv', index=False)
tst_data_scaled.to_csv('test.csv', index=False)

dataset = TSDataset(split='both', datasets_root=datasets_root,cont_vars=['Length',], cat_vars=['Stable'], lbl_as_feat=True)

ds = TSDataset(split='both', datasets_root=datasets_root,cont_vars=['Length',], cat_vars=['Stable'], lbl_as_feat=True)


trained_model =\
    VAE.load_from_checkpoint('../weights/vae_weights-v10.ckpt')
#trained_model.cuda() # перенос модели на графический вычислитель в случае использования GPU
trained_model.freeze()
#pred = model(x.cuda()) # перенос модели на графический вычислитель в случае использования GPU

losses = []
# run predictions for the training set examples
for i in range(len(dataset)):
    x_cont, x_cat = dataset[i]
    x_cont.unsqueeze_(0)
    x_cat.unsqueeze_(0)
#    recon, mu, logvar, x = trained_model.forward((x_cont.cuda(), x_cat.cuda())) # в случае использования GPU
    recon, mu, logvar, x = trained_model.forward((x_cont, x_cat))
    recon_loss, kld = trained_model.loss_function(x, recon, mu, logvar)
    losses.append(recon_loss + trained_model.hparams.kld_beta * kld)

data_with_losses_test = dataset.df
data_with_losses_test['loss'] = torch.asarray(losses)
#data_with_losses_test.sort_values('Шаг', inplace=True)

mean, sigma = data_with_losses_test['loss'].mean(), data_with_losses_test['loss'].std()

thresh = data_with_losses_test['loss'].quantile(quant)  # threshold для аномалий зависит от квантиля выборки (quant)

data_with_losses_test['anomaly'] = data_with_losses_test['loss'] > thresh

data_with_losses_unscaled_test = data_with_losses_test.copy()
#data_with_losses_unscaled_test[cont_vars] = scaler.inverse_transform(data_with_losses_test[cont_vars])
for enc, var in zip(label_encoders, cat_vars):
    data_with_losses_unscaled_test[var] = enc.inverse_transform(data_with_losses_test[var])
data_with_losses_unscaled_test = pd.DataFrame(data_with_losses_unscaled_test, columns=data_with_losses_test.columns)
print(data_with_losses_unscaled_test.head())
data_with_losses_unscaled_test.to_csv(output_file_path, index=False)

anomalies_value = data_with_losses_unscaled_test.loc[data_with_losses_unscaled_test['anomaly'], ['loss',
                                                                                                 'Length']]
normals_value = data_with_losses_unscaled_test.loc[~data_with_losses_unscaled_test['anomaly'], ['loss',
                                                                                                'Length']]