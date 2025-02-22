import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

####FUNCTIONS TO BE USED INSIDE THE METHODS
def valuesCenterer(df: np.ndarray) -> np.ndarray: 
    centered_df= df - df.mean(axis=0)
    return centered_df

def SVD(X: np.ndarray) -> np.ndarray:
        U,S,Vt= np.linalg.svd(X)
        return (U,S,Vt) 

def dataProjection(X:np.ndarray,pcMat:np.ndarray)-> np.ndarray:
    Y= X@pcMat
    dfnewpc= pd.DataFrame(Y,columns=['PC1','PC2']) 
    return dfnewpc.head()

def pcmatrixMaker(pc1: np.ndarray,pc2:np.ndarray)->np.ndarray:
    pcmat = np.column_stack((pc1, pc2))  
    return pcmat

def screeplot(num_components: list, variance_ratios_algo: list, variance_ratios_sklearn: list) -> None:
    """Plot scree plot for both custom PCA algorithm and sklearn PCA."""
    plt.figure(figsize=(8, 6))

    plt.plot(num_components,variance_ratios_algo, marker='o', linestyle='-', color='r', label='Scratch PCA',linewidth=0.5,markersize=3.5)
    plt.plot(num_components,variance_ratios_sklearn, marker='s', linestyle='--', color='b', label='Sklearn PCA',linewidth=0.5,markersize=3.5,alpha=0.5)

    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained Ratio')
    plt.title('Scree Plot Comparison')
    plt.xticks(num_components)
    plt.legend()
    plt.grid()
    plt.show()

####PCA USING COVARIANCE MATRIX EIG VALUES AND VECTORS - METHOD 1 IN THE PAPER 
class MethodOne: 

    def covMat(self,X: np.ndarray)-> np.ndarray: 
        vals= valuesCenterer(X)
        n= X.shape[0]
        return ((vals.T @ vals)*(1/(n-1)))

    def EigValsEigVecs(self,X: np.ndarray) -> tuple:
        eigvals,eigvecs= np.linalg.eig(X)
        eigen_dict= {eigvals[i]: eigvecs[:,i] for i in range(len(eigvals))}
        return eigen_dict

    def GetprincipalComps(self,eigdict: dict) -> np.ndarray:
        top_2_pc = [value for key, value in sorted(eigdict.items(), key=lambda item: item[0], reverse=True)]
        top_keys = sorted(eigdict.keys(), reverse=True)[:2]
        keysum= sum(eigdict.keys())

        return (top_2_pc[0],top_2_pc[1],top_keys[0]/keysum,top_keys[1]/keysum)
    


####PCA USING SVD, A MORE GENERALIZED APPROACH, METHOD 2 IN THE PAPER 
class MethodTwo:

    def SVDApply(self,df:pd.DataFrame) -> np.ndarray: 
        vals= df.values
        vals= valuesCenterer(df)
        U,S,Vt = SVD(vals)
        return (Vt[0,:],Vt[1,:])

