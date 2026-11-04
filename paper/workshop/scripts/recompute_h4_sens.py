# RECOMPUTED: H4 slope (mean over seeds of per-seed OLS slope of Da-Db in s), formal vs sensitivity (a) (failed episodes with rmse<=1 m scored raw).
# also slope using only s>=0.5 (endpoint s=0 removed) for circle.
import json,numpy as np,collections
E=collections.defaultdict(dict)
for l in open('results/prereg/episodes.jsonl'):
    r=json.loads(l)
    if r['speed']!=1.0: continue
    if r['method']=='CON-ATT' and r['hp'].get('p_drop',0)!=0: continue
    fin=r['finite'] and r['rmse']<=1
    formal=100.0 if (not fin or not r['success']) else 100*r['rmse']
    sens=100*r['rmse'] if fin else 100.0
    E[(r['method'],r['profile'],r['traj'])][r['seed']]=(formal,sens)
S=np.array([0,0.5,1,1.75,2.5])
def arr(m,s,t,i):
    d=E[(m,f's={s:g}',t)]; return np.array([d[k][i] for k in sorted(d)])
def slope(x,Y): xc=x-x.mean(); return ((xc[:,None]*Y).sum(0)/(xc**2).sum())
for t in ['circle','fig8']:
    for name,i in [('formal',0),('sens_a',1)]:
        Y=np.array([(arr('EKF-L+GEO',s,t,i)-arr('EKF-L',s,t,i))-(arr('CON-ATT',s,t,i)-arr('EKF-L+GEO',s,t,i)) for s in S])
        print(t,name,'H4 slope all s: %.3f'%slope(S,Y).mean(),' s>=0.5 only: %.3f'%slope(S[1:],Y[1:]).mean(),
              ' Da-Db at s: ',np.round(Y.mean(1),2))
