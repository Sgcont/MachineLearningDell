import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


data = pd.read_table("fruit_data_with_colors_miss.txt", na_values=[".", "?"])

numericas = data.select_dtypes(include=["number"]).columns
data[numericas] = data[numericas].fillna(data[numericas].mean())
data["fruit_subtype"] = data["fruit_subtype"].fillna(data["fruit_subtype"].mode(dropna=True).iloc[0])

features = data[["mass", "width", "height", "color_score"]].copy()
scaler = StandardScaler()
features_esc = scaler.fit_transform(features)

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(features_esc)

svd = TruncatedSVD(n_components=2, random_state=42)
svd_result = svd.fit_transform(features_esc)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_esc)

print("Primeiras linhas dos dados tratados:")
print(data.head())

print("\nPCA:")
print(pd.DataFrame(pca_result, columns=["PCA1", "PCA2"]).head())

print("\nSVD:")
print(pd.DataFrame(svd_result, columns=["SVD1", "SVD2"]).head())

print("\nK-means:")
print(pd.Series(clusters, name="cluster").value_counts().sort_index())

print("\nVariância explicada pelo PCA:")
print(pca.explained_variance_ratio_)
