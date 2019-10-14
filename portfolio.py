from pyomo.environ import *
import pandas as pd
# import data
array_df = pd.read_table('prt_array_data.txt', sep='\t')
cov_df = pd.read_table('covariance_data.txt', index_col=0, sep='\t')
# data in pyomo input format
investment_strs = array_df['tickers'].tolist()
return_floats = array_df['returns'].tolist()
ubounds = array_df['ubounds'].tolist()
lbounds = array_df['lbounds'].tolist()
# user defined constraints
net_upper_long_bound=1.0
net_lower_long_bound=.5
net_upper_short_bound=1.0
net_lower_short_bound=.5
# cardinality limit
card = 15
# risk/reward tradeoff input, higher means less risk, b/t .1 & 1
rho = .5
# model initialization
m = AbstractModel()
m.investments = Set(initialize=investment_strs)
ret_dict = dict(zip(investment_strs, return_floats))
m.returns = Param(m.investments, initialize=ret_dict)
ub_dict = dict(zip(investment_strs, ubounds))
lb_dict = dict(zip(investment_strs, lbounds))
m.upper_asset_bounds = Param(m.investments, initialize=ub_dict)
m.lower_asset_bounds = Param(m.investments, initialize=lb_dict)
# covariance initialization
def cov_init(m, i, j):    
    return cov_df.loc[i, j]
m.covariance_mat = Param(m.investments, m.investments, initialize=cov_init)
# decision variable initialization
# allocation to short and long investments
m.allocation_s = Var(m.investments, domain=NonNegativeReals)
m.allocation_l = Var(m.investments, domain=NonNegativeReals)
# binary on whether investment has a short or long allocation
m.d_l = Var(m.investments, domain=Boolean)
m.d_s = Var(m.investments, domain=Boolean)
# objective and constraint functions:
def obj_rule(m):
    """
    objective function: total expected return minus risk, 
    with rho serving as user defined weighting on tradeoff
    utility (b/t 0.1 and 1)
    """
    return sum(
            m.returns[i]* (m.allocation_l[i] - m.allocation_s[i])
                for i in m.investments
                ) - \
        rho * sum(
            m.covariance_mat[i,j]*(
                    m.allocation_l[i]-m.allocation_s[i]
                    ) *
                (m.allocation_l[j]-m.allocation_s[j]) 
                    for (i,j) in m.covariance_mat
                )
def lflag_rule(m, i):
    """constraint on long investments below their upper bound
    """
    return m.allocation_l[i] <= m.d_l[i] * m.upper_asset_bounds[i]
def sflag_rule(m, i):
    """constraint on short investments above their lower bound
    """
    return -m.allocation_s[i]>=m.d_s[i] * m.lower_asset_bounds[i]
def card_rule(m):
    """constraint on total number of investments below 
    cardinality input
    """
    return sum((m.d_l[i] + m.d_s[i]) for i in m.investments)<=card
def binary_rule(m, i):
    """constraint that an individual ticker be only long, 
    short or neither
    """
    return m.d_l[i] + m.d_s[i] <= 1
def netlong_rule_ub(m, i):
    """constraint on upper bound aggregate long investment
    """
    return sum((m.allocation_l[i]) for i in m.investments) <= net_upper_long_bound
def netlong_rule_lb(m, i):
    """constraint on lower bound aggregate long investment
    """
    return sum((m.allocation_l[i]) for i in m.investments) <= \
        net_upper_long_bound
def netshrt_rule_ub(m, i):
    """constraint on upper bound aggregate shrt investment
    """
    return sum((m.allocation_s[i]) for i in m.investments) <= \
        net_upper_short_bound
def netshrt_rule_lb(m, i):
    """constraint on lower bound aggregate shrt investment
    """
    return sum((m.allocation_s[i]) for i in m.investments) >= \
        net_lower_short_bound
# set model object and constraint functions:
m.OBJ = Objective(rule=obj_rule, sense=maximize)
m.cons_long_invest_ubs = Constraint(m.investments, rule=lflag_rule)
m.cons_shrt_invest_lbs = Constraint(m.investments, rule=sflag_rule)
m.cons_cardinality_ubs = Constraint(rule=card_rule)
m.cons_long_shrt_none = Constraint(m.investments, rule=binary_rule)
m.net_long_upper = Constraint(m.investments, rule=netlong_rule_ub)
m.net_long_lower = Constraint(m.investments, rule=netlong_rule_lb)
m.net_short_upper = Constraint(m.investments, rule=netshrt_rule_ub)
m.net_short_lower = Constraint(m.investments, rule=netshrt_rule_lb)
# run model
opt = SolverFactory("scipampl")
m.construct()
results = opt.solve(m)
# view results
rs = sum(m.returns[i]* ( m.allocation_l[i].value - m.allocation_s[i].value ) for i in m.investments)
covs = sum(m.covariance_mat[i,j] * (m.allocation_l[i].value - m.allocation_s[i].value)*(m.allocation_l[j].value-m.allocation_s[j].value) for (i,j) in m.covariance_mat)
cards = sum(m.d_l[i].value+m.d_s[i].value for i in m.investments)
tlong = sum(m.d_l[i].value*m.allocation_l[i].value for i in m.investments)
clong = sum(m.d_l[i].value for i in m.investments)
tshrt = sum(m.d_s[i].value*m.allocation_s[i].value for i in m.investments)
cshrt = sum(m.d_s[i].value for i in m.investments)
ann_sharpe = pd.np.sqrt(252)*rs / pd.np.sqrt(covs)
print('aggregate expected return: daily: %.4f, ann: %.3f' % (rs, rs*252))
print('aggregate vol return: daily: %.4f, ann: %.3f' % (pd.np.sqrt(covs), pd.np.sqrt(covs*252)))
print('annualized sharpe: %.4f' % ann_sharpe)
print('total number of names: ', cards)
print('total number of longs: ', clong)
print('total allocation to longs: %.2f' % tlong)
print('total number of shrts: ', cshrt)
print('total allocation to shrts: %.2f' % tshrt)
print('allocations:')
print('ticker', ' long'.ljust(5), ' short'.ljust(5))
for i in m.investments:
    print(i.ljust(6), ' %.3f' % m.allocation_l[i].value, ' %.3f' % m.allocation_s[i].value)
