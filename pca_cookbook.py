from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib.pyplot as plt 

class PCACook: 
    def pcafit(self,X:pd.DataFrame)->None:
        pca = PCA(n_components=2)
        X_pca = pca.fit(X)
        X_transofrmed = pca.fit_transform(X)
        df_pca= pd.DataFrame(X_transofrmed,columns=['PC1','PC2'])
        return X_pca.components_, df_pca.head(), X_pca.explained_variance_ratio_
    






