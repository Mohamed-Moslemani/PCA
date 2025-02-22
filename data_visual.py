import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your PCA modules (do not rewrite their functions)
from pca import dataProjection, MethodOne, pcmatrixMaker, valuesCenterer
from pca_cookbook import PCACook

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


method_one = MethodOne()
cov_matrix = method_one.covMat(iris.data)
eig_dict= method_one.EigValsEigVecs(cov_matrix)
pc1, pc2, var1, var2 = method_one.GetprincipalComps(eig_dict)
pc_mat= pcmatrixMaker(pc1, pc2)
X_centered= valuesCenterer(iris.data)

df_scratch_pca= dataProjection(X_centered, pc_mat)
df_scratch_pca['species'] = df_iris['species']


pca_cook = PCACook()
components, df_sklearn_pca, explained_variance = pca_cook.pcafit(df_iris[iris.feature_names])
df_sklearn_pca['species'] = df_iris['species']


color_map = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors_orig = df_iris['species'].map(color_map).tolist()
colors_scratch = df_scratch_pca['species'].map(color_map).tolist()
colors_sklearn = df_sklearn_pca['species'].map(color_map).tolist()


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Scratch PCA", "Sklearn PCA")
)

fig.add_trace(
    go.Scatter(
        x=df_iris[iris.feature_names[0]],
        y=df_iris[iris.feature_names[1]],
        mode='markers',
        marker=dict(color=colors_orig, size=8),
        text=df_iris['species'],
        name="Original Features",
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_scratch_pca['PC1'],
        y=df_scratch_pca['PC2'],
        mode='markers',
        marker=dict(color=colors_scratch, size=8),
        text=df_scratch_pca['species'],
        name="PCA Components"
    ),
    row=1, col=1
)


fig.add_trace(
    go.Scatter(
        x=df_iris[iris.feature_names[0]],
        y=df_iris[iris.feature_names[1]],
        mode='markers',
        marker=dict(color=colors_orig, size=8),
        text=df_iris['species'],
        name="Original Features"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df_sklearn_pca['PC1'],
        y=df_sklearn_pca['PC2'],
        mode='markers',
        marker=dict(color=colors_sklearn, size=8),
        text=df_sklearn_pca['species'],
        name="PCA Components"
    ),
    row=1, col=2
)

fig.data[0].visible = True   
fig.data[1].visible = False  
fig.data[2].visible = True   
fig.data[3].visible = False  


fig.update_layout(
    title="Iris Data: Original Features vs. PCA Components",
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            buttons=[
                dict(
                    label="Original Features",
                    method="update",
                    args=[{"visible": [True, False, True, False]},
                          {"title": "Iris Data: Original Features"}]
                ),
                dict(
                    label="PCA Components",
                    method="update",
                    args=[{"visible": [False, True, False, True]},
                          {"title": "Iris Data: PCA Components"}]
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top"
        )
    ],
    width=1000,
    height=500,
    showlegend=False
)


fig.show()
