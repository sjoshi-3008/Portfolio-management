import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.patches import Shadow

style.use('ggplot')
df=pd.read_csv('smport.csv', index_col='Date', parse_dates=True)
print(df.head(10))
df.info()
print(df.describe())

from pypfopt  import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
# Calculating expected returns mu 
mu = expected_returns.mean_historical_return(df)

# Calculating the covariance matrix S
Sigma = risk_models.sample_cov(df)

# Obtaining the efficient frontier
ef = EfficientFrontier(mu, Sigma)
print (mu, Sigma)
returns=df.pct_change()
covMatrix = returns.cov()*251
print(covMatrix)
# Getting the minimum risk portfolio for a target return 
weights = ef.efficient_return(0.2)
print (weights)
l=list(df.columns)
print(l)
size=list(weights.values())
print(size)
print(type(size))
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Return=20%')
plt.show()
# Showing portfolio performance 
ef.portfolio_performance(verbose=True)

# Calculating weights for the maximum Sharpe ratio portfolio
raw_weights_maxsharpe = ef.max_sharpe()
cleaned_weights_maxsharpe = ef.clean_weights()
print (raw_weights_maxsharpe, cleaned_weights_maxsharpe)
ef.portfolio_performance(verbose=True)
size=list(cleaned_weights_maxsharpe.values())
print(size)
import matplotlib as mpl
mpl.rcParams['font.size'] = 20.0
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0.0, 0.0, 0.8, 0.8])
explode = (0, 0, 0.05, 0, 0, 0)
pies=ax.pie(size, explode=explode,labels=l, labeldistance=1.2, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16})
plt.title("Markovitz' portfolio with maximum return", fontsize=36)
for w in pies[0]:
    # set the id with the label.
    w.set_gid(w.get_label())

    # we don't want to draw the edge of the pie
    w.set_edgecolor("none")

for w in pies[0]:
    # create shadow patch
    s = Shadow(w, -0.01, -0.01)
    s.set_gid(w.get_gid() + "_shadow")
    s.set_zorder(w.get_zorder() - 0.1)
    ax.add_patch(s)
plt.show()
fig.savefig("Markovitz' portfolio with maximum return.png")


# Calculating weights for the minimum volatility portfolio
raw_weights_minvol = ef.min_volatility()
cleaned_weights_minvol = ef.clean_weights()

# Showing portfolio performance
print(cleaned_weights_minvol)
ef.portfolio_performance(verbose=True)
size=list(cleaned_weights_minvol.values())
print(size)
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Min Risk')
plt.show()

#Calculating an exponentially weighted portfolio
Sigma_ew = risk_models.exp_cov(df, span=180, frequency=252)
mu_ew = expected_returns.ema_historical_return(df, frequency=252, span=180)
# Calculate the efficient frontier
ef_ew = EfficientFrontier(mu_ew, Sigma_ew)
# Calculate weights for the maximum sharpe ratio optimization
raw_weights_maxsharpe_ew = ef_ew.max_sharpe()
# Show portfolio performance 
ef_ew.portfolio_performance(verbose=True)
size=list(raw_weights_maxsharpe_ew.values())
print(size)
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Max Return EW')
plt.show()
