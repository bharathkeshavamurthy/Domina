"""
EEE551 | Information Theory | Homework-I | Renyi Entropy
"""

# The imports
import plotly
import numpy as np
import plotly.graph_objs as go

# User Credentials
plotly.tools.set_credentials_file(username='bkeshava_cisco', api_key='z5ugUalukC19VzAh3uVw')


# Core Function
def renyi_entropy(alpha):
    return (1 / (1 - alpha)) * np.log2(0.1 ** alpha + 0.9 ** alpha)


alphas = np.concatenate((np.arange(start=0.0, stop=1.0, step=0.01, dtype=np.float64),
                         np.arange(start=1.01, stop=100.01, step=0.01, dtype=np.float64)), axis=0)
trace = go.Scatter(x=alphas, y=np.array([renyi_entropy(_alpha) for _alpha in alphas]), mode='lines+markers')
layout = dict(title='Renyi Entropy vs Alpha for a Bernoulli Random Variable with p=0.1',
              xaxis=dict(title='Alpha', autorange=True),
              yaxis=dict(title='Renyi Entropy', autorange=True))
fig = dict(data=[trace], layout=layout)
fig_url = plotly.plotly.plot(fig, filename='Renyi_Entropy', auto_open=False)
print(fig_url)
