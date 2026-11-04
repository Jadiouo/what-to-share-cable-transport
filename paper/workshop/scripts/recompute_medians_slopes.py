import json,numpy as np,collections
E=collections.defaultdict(dict)
for l in open('results/prereg/episodes.jsonl'):
    r=json.loads(l)
    if r['speed']!=1.0: continue
    pd=r['hp'].get('p_drop',0) if r['method']=='CON-ATT' else 0
    if r['method']=='CON-ATT' and pd!=0: continue
    eff=100.0 if (not r['finite'] or not r['success'] or r['rmse']>1) else 100*r['rmse']
    E[(r['method'],r['profile'],r['traj'])][r['seed']]=(eff,100*r['rmse'] if r['finite'] else np.nan,r['success'])
S=[0,0.5,1,1.75,2.5]
def arr(m,s,t,i): 
    d=E[(m,f's={s:g}',t)]; return np.array([d[k][i] for k in sorted(d)])
for t in ['circle','fig8']:
    print(t)
    for s in S:
        a=arr('EKF-L+GEO',s,t,0)-arr('EKF-L',s,t,0); b=arr('CON-ATT',s,t,0)-arr('EKF-L+GEO',s,t,0)
        print(s,'H1a mean/med %.2f %.2f'%(a.mean(),np.median(a)),'H1b mean/med %.2f %.2f'%(b.mean(),np.median(b)), 'n',len(a))
    # raw slope
    x=np.array(S);
    for name,i in [('eff',0),('raw',1)]:
        sl=[]
        D=np.array([arr('DOB',s,t,i)-arr('CON-ATT',s,t,i) for s in S]) # 5x40
        sl=((x-x.mean())[:,None]*D).sum(0)/((x-x.mean())**2).sum()
        print(name,'H2 slope mean',sl.mean(),(sl>0).sum(),len(sl))
    # median slopes eff
    D=np.array([arr('DOB',s,t,0)-arr('CON-ATT',s,t,0) for s in S]);sl=((x-x.mean())[:,None]*D).sum(0)/3.95
    print('eff slope median',np.median(sl))
# DOB-CON raw mean at s
for t in ['circle','fig8']:
    print(t,[round(float((arr('DOB',s,t,1)-arr('CON-ATT',s,t,1)).mean()),2) for s in S])
    print('raw median',[round(float(np.median(arr('DOB',s,t,1)-arr('CON-ATT',s,t,1))),2) for s in S])
