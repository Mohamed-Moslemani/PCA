from pca_cookbook import PCACook
from pca import MethodOne, MethodTwo,pcmatrixMaker,dataProjection, valuesCenterer,screeplot
from sklearn.datasets import load_iris 
import pandas as pd 

m1= MethodOne()
m2= MethodTwo()
pcasklrn= PCACook()

df_iris = load_iris()
df = pd.DataFrame(df_iris.data,columns=df_iris.feature_names)
X= df.values 
X= valuesCenterer(df.values)
covmatr= m1.covMat(df)
eigsvecsvals= m1.EigValsEigVecs(covmatr)
pccomps11,pcomps12,var1,var2= m1.GetprincipalComps(eigsvecsvals)
print("principal components using method 1: ")
print(pccomps11)
print(pcomps12)
print("=========================================================================================================================")
print("=========================================================================================================================")
pcomps21,pcomps22= m2.SVDApply(df)
print("Principal components using method 2: ")
print(pcomps21)
print(pcomps22)
print("Variance ratio of PC1: ")
print(var1)
print("Variance ratio of PC2: ")
print(var2)
print("=========================================================================================================================")
print("=========================================================================================================================")
pccompsCookbook,projectedDataSk,sklearnVarRatio= pcasklrn.pcafit(df)
print("Principal components using sklearn: ")
print(pccompsCookbook)
print("Sklearn variance ratio: ")
print(sklearnVarRatio)
print("-------------------------------------------------------------------------------------------------------------------------")
print("=========================================================================================================================")
print("=========================================================================================================================")
print("-------------------------------------------------------------------------------------------------------------------------")
pcmat1= pcmatrixMaker(pccomps11,pcomps12)
pcmat2= pcmatrixMaker(pcomps21,pcomps22)
print("Data projection using method 1: ")
print(dataProjection(pcMat=pcmat1,X=X))
print("=========================================================================================================================")
print("=========================================================================================================================")
print("Data projection using Sklearn: ")
print(projectedDataSk)

variance_ratios_algo = [var1, var2]
variance_ratios_sklearn= list(sklearnVarRatio)
num_components = range(1,len(variance_ratios_sklearn) + 1)

screeplot(num_components, variance_ratios_algo, variance_ratios_sklearn)

