from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline


# Get X and Y datasets
x, y = tratamento_dados_empresa.get_x_y(x_csv_path, y_csv_path)

# Feature Scaling X
x_scaler = StandardScaler()
x = x_scaler.fit_transform(x)

# Feature Scaling y
# y_scaler = StandardScaler()
# y = y_scaler.fit_transform(y)


pca = PCA(n_components=1)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns=['pca1'])


plt.plot(y, principalDf['pca1'], 'ro')
plt.show()

pca.explained_variance_ratio_

pca.explained_variance_ratio_.sum()

principalDf