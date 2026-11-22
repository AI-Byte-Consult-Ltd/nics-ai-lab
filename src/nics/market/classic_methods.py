"""Ten well-known trading methods as code rules, plus our own methods ported to daily bars, and
an honest daily backtester. Research only (scripts/methods_research.py); nothing here trades.

Own parameters, no book text, no names of authors, schools or companies. Decisions use the
close of bar i only and are executed at the open of bar i+1; stops are checked inside the bar
(a gap through the stop exits at the open), the stop is checked before the target on the same
bar, costs = 1.2 x median broker spread + swap per calendar day held, and a position still open
at the end is marked at the last close. Results: docs/methods_research_2026-11-22.md.
"""
from __future__ import annotations

import numpy as np

# ---------- indicators (local copies so the research stays reproducible) ----------
def sma(x,n):
    o=np.full(len(x),np.nan); 
    if len(x)>=n:
        c=np.cumsum(np.insert(x,0,0.0)); o[n-1:]=(c[n:]-c[:-n])/n
    return o
def ema(x,n):
    o=np.empty(len(x)); k=2/(n+1); a=x[0]
    for i,v in enumerate(x):
        a=v*k+a*(1-k) if i else v; o[i]=a
    o[:n-1]=np.nan; return o
def wilder(x,n):
    o=np.full(len(x),np.nan)
    if len(x)<n: return o
    a=np.mean(x[:n]); o[n-1]=a
    for i in range(n,len(x)):
        a=(a*(n-1)+x[i])/n; o[i]=a
    return o
def tr(h,l,c):
    pc=np.concatenate([[c[0]],c[:-1]]); return np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)))
def atr(h,l,c,n=14): return wilder(tr(h,l,c),n)
def rsi(c,n):
    d=np.diff(c,prepend=c[0]); up=wilder(np.maximum(d,0),n); dn=wilder(np.maximum(-d,0),n)
    with np.errstate(divide='ignore',invalid='ignore'):
        r=100-100/(1+up/dn)
    r=np.where((dn==0)&(up>0),100.0,r); r=np.where((dn==0)&(up==0),50.0,r); return r
def rmax(x,n):
    o=np.full(len(x),np.nan)
    if len(x)>=n: o[n-1:]=np.lib.stride_tricks.sliding_window_view(x,n).max(1)
    return o
def rmin(x,n):
    o=np.full(len(x),np.nan)
    if len(x)>=n: o[n-1:]=np.lib.stride_tricks.sliding_window_view(x,n).min(1)
    return o
def rstd(x,n):
    o=np.full(len(x),np.nan)
    if len(x)>=n: o[n-1:]=np.lib.stride_tricks.sliding_window_view(x,n).std(1)
    return o
def shift(x,k=1):
    o=np.full(len(x),np.nan); o[k:]=x[:-k]; return o
def adx(h,l,c,n=14):
    up=np.diff(h,prepend=h[0]); dn=-np.diff(l,prepend=l[0])
    p=np.where((up>dn)&(up>0),up,0.0); m=np.where((dn>up)&(dn>0),dn,0.0)
    a=wilder(tr(h,l,c),n)
    with np.errstate(divide='ignore',invalid='ignore'):
        pdi=100*wilder(p,n)/a; mdi=100*wilder(m,n)/a; dx=100*abs(pdi-mdi)/(pdi+mdi)
    dx=np.nan_to_num(dx,nan=0.0); o=np.full(len(c),np.nan); s=2*n-1
    if len(c)>s:
        v=np.mean(dx[n:s+1]) if s+1>n else 0; o[s]=v
        for i in range(s+1,len(c)): v=(v*(n-1)+dx[i])/n; o[i]=v
    return o
def supertrend(h,l,c,n=10,k=3.0):
    a=atr(h,l,c,n); hl2=(h+l)/2; ub=hl2+k*a; lb=hl2-k*a
    fu=np.full(len(c),np.nan); fl=np.full(len(c),np.nan); d=np.zeros(len(c),int)
    for i in range(len(c)):
        if np.isnan(a[i]): continue
        if i==0 or np.isnan(fu[i-1]): fu[i]=ub[i]; fl[i]=lb[i]; d[i]=1; continue
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        d[i]= 1 if c[i]>fu[i-1] else (-1 if c[i]<fl[i-1] else d[i-1])
    line=np.where(d>0,fl,fu); return d,line



def base(d):
    o,h,l,c=d['o'],d['h'],d['l'],d['c']; n=len(c)
    return o,h,l,c,n,atr(h,l,c,14)

def _cross(a,b):
    up=(a>b)&(shift(a)<=shift(b)); dn=(a<b)&(shift(a)>=shift(b)); return up,dn

# ---------- new methods (neutral ids) ----------
def m01_ma_cross(d, f=50, s=200, k=3.0):
    o,h,l,c,n,a=base(d); ef,es=ema(c,f),ema(c,s); up,dn=_cross(ef,es)
    ent=np.where(up,1,np.where(dn,-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=dn, exit_short=up)

def m02_channel_55_20(d, ne=55, nx=20, k=2.0):
    o,h,l,c,n,a=base(d); he,le=shift(rmax(h,ne)),shift(rmin(l,ne)); hx,lx=shift(rmax(h,nx)),shift(rmin(l,nx))
    ent=np.where(c>he,1,np.where(c<le,-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=c<lx, exit_short=c>hx)

def m03_rsi2_pullback(d, lo=10, hi=90, k=3.0, mx=10):
    o,h,l,c,n,a=base(d); r=rsi(c,2); s200=sma(c,200); s5=sma(c,5)
    ent=np.where((c>s200)&(r<lo),1,np.where((c<s200)&(r>hi),-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=c>s5, exit_short=c<s5, max_bars=mx)

def m04_band_reversion(d, nb=20, z=2.0, k=2.0, mx=15):
    o,h,l,c,n,a=base(d); m=sma(c,nb); sd=rstd(c,nb)
    ent=np.where(c<m-z*sd,1,np.where(c>m+z*sd,-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=c>=m, exit_short=c<=m, max_bars=mx)

def m05_macd_trend(d, k=2.5):
    o,h,l,c,n,a=base(d); mac=ema(c,12)-ema(c,26); sig=np.full(n,np.nan)
    v=~np.isnan(mac); sig[v]=ema(mac[v],9)
    up,dn=_cross(mac,sig)
    ent=np.where(up&(mac>0),1,np.where(dn&(mac<0),-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=dn, exit_short=up)

def m06_nr7_breakout(d, rr=2.0, mx=5):
    o,h,l,c,n,a=base(d); rg=h-l; nr=rg<=rmin(rg,7)
    # pending OCO stop orders for the next bar
    return dict(oco=nr, buy_px=h+0.05*a, sell_px=l-0.05*a, buy_sl=l-0.05*a, sell_sl=h+0.05*a, rr=rr, max_bars=mx)

def m07_tsmom(d, lb=252, k=3.0):
    o,h,l,c,n,a=base(d); t=d['t']
    mon=np.array([np.datetime64(int(x),'s').astype('datetime64[M]').astype(int) for x in t])
    last=np.r_[mon[1:]!=mon[:-1],False]
    ret=c/shift(c,lb)-1; sgn=np.sign(np.nan_to_num(ret))
    ent=np.where(last&(sgn>0),1,np.where(last&(sgn<0),-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=last&(sgn<0), exit_short=last&(sgn>0))

def m08_atr_trail(d, nn=10, kk=3.0):
    o,h,l,c,n,a=base(d); dr,line=supertrend(h,l,c,nn,kk)
    pd=shift(dr.astype(float)); up=(dr==1)&(pd==-1); dn=(dr==-1)&(pd==1)
    ent=np.where(up,1,np.where(dn,-1,0))
    return dict(ent=ent, stop_px=line, trail=line, exit_long=dn, exit_short=up)

def m09_cloud(d, k=2.5):
    o,h,l,c,n,a=base(d)
    ten=(rmax(h,9)+rmin(l,9))/2; kij=(rmax(h,26)+rmin(l,26))/2
    sa=shift((ten+kij)/2,26); sb=shift((rmax(h,52)+rmin(l,52))/2,26)
    top=np.fmax(sa,sb); bot=np.fmin(sa,sb); up,dn=_cross(ten,kij)
    ent=np.where(up&(c>top),1,np.where(dn&(c<bot),-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=c<kij, exit_short=c>kij)

def m10_squeeze(d, nb=20, look=120, k=2.0, mx=40):
    o,h,l,c,n,a=base(d); m=sma(c,nb); sd=rstd(c,nb); bw=4*sd/m
    sq=bw<=rmin(bw,look)*1.0001; recent=rmax(sq.astype(float),5)>0
    ent=np.where(recent&(c>m+2*sd),1,np.where(recent&(c<m-2*sd),-1,0))
    return dict(ent=ent, stop_dist=k*a, exit_long=c<m, exit_short=c>m, max_bars=mx)

# ---------- ours, portable to D1 (same rules as in NICS training / swing shadow) ----------
def o1_range_both_sides(d):
    o,h,l,c,n,a=base(d); ax=adx(h,l,c); hi,lo=shift(rmax(h,48)),shift(rmin(l,48)); mid=(hi+lo)/2
    ok=(ax<20)&(hi-lo>=3*a)
    L=ok&(l<=lo+0.1*a)&(c>lo); S=ok&~L&(h>=hi-0.1*a)&(c<hi)
    return dict(ent=np.where(L,1,np.where(S,-1,0)), stop_px=np.where(L,lo-0.6*a,hi+0.6*a), target_px=mid, max_bars=20)

def o2_breakout20(d):
    o,h,l,c,n,a=base(d); ax=adx(h,l,c); hi,lo=shift(rmax(h,20)),shift(rmin(l,20))
    ent=np.where((ax>20)&(c>hi),1,np.where((ax>20)&(c<lo),-1,0))
    return dict(ent=ent, stop_dist=1.5*a, target_dist=2.0*a, max_bars=20)

def o3_ema_pullback(d):
    o,h,l,c,n,a=base(d); e20,e50=ema(c,20),ema(c,50); e50p=shift(e50,5)
    L=(e20>e50)&(e50>e50p)&(l<=e20)&(e20<c); S=(e20<e50)&(e50<e50p)&(h>=e20)&(e20>c)
    sl=np.where(L,rmin(l,5)-0.5*a,rmax(h,5)+0.5*a)
    return dict(ent=np.where(L,1,np.where(S,-1,0)), stop_px=sl, target_r=1.5, max_bars=20)

def o4_swing_d20(d):
    # [30 SWING] D20_BREAKOUT_TREND_v1 rules on daily bars: 20-day range breakout with EMA20 vs EMA50 trend, stop 1 ATR, TP 2R, hold <= 2 days
    o,h,l,c,n,a=base(d); e20,e50=ema(c,20),ema(c,50); hi,lo=shift(rmax(h,20)),shift(rmin(l,20))
    ent=np.where((c>hi)&(e20>e50),1,np.where((c<lo)&(e20<e50),-1,0))
    return dict(ent=ent, stop_dist=1.0*a, target_r=2.0, max_bars=2)

NEW={'m01_ma_cross':m01_ma_cross,'m02_channel_55_20':m02_channel_55_20,'m03_rsi2_pullback':m03_rsi2_pullback,
     'm04_band_reversion':m04_band_reversion,'m05_macd_trend':m05_macd_trend,'m06_nr7_breakout':m06_nr7_breakout,
     'm07_tsmom_12m':m07_tsmom,'m08_atr_trail':m08_atr_trail,'m09_cloud':m09_cloud,'m10_squeeze':m10_squeeze}
OURS={'o1_range_both_sides':o1_range_both_sides,'o2_breakout20':o2_breakout20,'o3_ema_pullback':o3_ema_pullback,'o4_swing_d20':o4_swing_d20}


# ---------- simulator ----------
def simulate(d, M, cost_rel, swap_rate, warm=260, trade_from=None, bar_days=1.0):
    """Event loop. Decisions at close of bar i, executed at open of bar i+1. Stops intrabar (gap -> open).
    Stop is checked before target on the same bar. Returns list of trades."""
    o,h,l,c,t=d['o'],d['h'],d['l'],d['c'],d['t']; n=len(c)
    a=atr(h,l,c,14)
    g=lambda k: M.get(k)
    ent=g('ent'); exL=g('exit_long'); exS=g('exit_short'); mx=g('max_bars'); trail=g('trail')
    trades=[]; pos=None; pend=None; pexit=False; oco=None
    def close_pos(j,px,why):
        nonlocal pos
        dd=pos['dir']; e=pos['entry']; risk=pos['risk']
        days=max((t[j]-t[pos['bar']])/86400.0,0.0)+ (0 if bar_days<1 else 0)
        cost=cost_rel*e + swap_rate*e*days/365.0
        r_g=dd*(px-e)/risk; r=(dd*(px-e)-cost)/risk
        trades.append(dict(t_in=int(t[pos['bar']]),t_out=int(t[j]),dir=dd,r=r,r_gross=r_g,cost_r=cost/risk,bars=j-pos['bar'],why=why))
        pos=None
    def open_pos(j,dd,e,i_sig,stop=None):
        nonlocal pos
        ai=a[i_sig]
        if stop is None:
            if g('stop_dist') is not None: stop=e-dd*M['stop_dist'][i_sig]
            else: stop=M['stop_px'][i_sig]
        if np.isnan(stop) or np.isnan(ai): return
        if dd*(e-stop)<0.5*ai: stop=e-dd*0.5*ai  # never closer than 0.5 ATR (gap through the planned stop)
        risk=dd*(e-stop)
        tgt=None
        if g('target_px') is not None: tgt=M['target_px'][i_sig]
        elif g('target_dist') is not None: tgt=e+dd*M['target_dist'][i_sig]
        elif g('target_r') is not None: tgt=e+dd*M['target_r']*risk
        if tgt is not None and (np.isnan(tgt) or dd*(tgt-e)<=0): tgt=None
        pos=dict(dir=dd,entry=e,stop=stop,risk=risk,tgt=tgt,bar=j,held=0)
    start=warm if trade_from is None else max(warm,int(np.searchsorted(t,trade_from)))
    for j in range(start,n):
        # ---- open of bar j
        if pos and pexit: close_pos(j,o[j],'signal'); pexit=False
        if pend is not None and pos is None:
            dd,i_sig=pend; open_pos(j,dd,o[j],i_sig)
        pend=None
        if oco is not None and pos is None:
            i_sig=oco; bp,sp=M['buy_px'][i_sig],M['sell_px'][i_sig]
            hitb=h[j]>=bp; hits=l[j]<=sp
            if hitb and hits:  # both touched: the one nearer the open first
                hits = (o[j]-sp) < (bp-o[j]); hitb = not hits
            if hitb:
                e=max(o[j],bp); st=M['buy_sl'][i_sig]; open_pos(j,1,e,i_sig,st)
                if pos: pos['tgt']=e+M['rr']*pos['risk']
            elif hits:
                e=min(o[j],sp); st=M['sell_sl'][i_sig]; open_pos(j,-1,e,i_sig,st)
                if pos: pos['tgt']=e-M['rr']*pos['risk']
        oco=None
        # ---- intrabar
        if pos:
            dd=pos['dir']; st=pos['stop']; tg=pos['tgt']
            if dd>0:
                if l[j]<=st: close_pos(j,min(st,o[j]) if pos['bar']<j else st,'stop')
                elif tg is not None and h[j]>=tg: close_pos(j,max(tg,o[j]) if pos['bar']<j else tg,'target')
            else:
                if h[j]>=st: close_pos(j,max(st,o[j]) if pos['bar']<j else st,'stop')
                elif tg is not None and l[j]<=tg: close_pos(j,min(tg,o[j]) if pos['bar']<j else tg,'target')
        if j>=n-1: break
        # ---- close of bar j
        if pos:
            pos['held']+=1; dd=pos['dir']
            if trail is not None and not np.isnan(trail[j]):
                pos['stop']=max(pos['stop'],trail[j]) if dd>0 else min(pos['stop'],trail[j])
            ex=(dd>0 and exL is not None and exL[j]) or (dd<0 and exS is not None and exS[j])
            if ex or (mx and pos['held']>=mx): pexit=True
        if (pos is None or pexit):
            if ent is not None and ent[j]!=0 and not (pos and not pexit):
                if pos is None or ent[j]!=pos['dir']: pend=(int(ent[j]),j)
            if g('oco') is not None and M['oco'][j] and pos is None: oco=j
    if pos: close_pos(n-1,c[n-1],'open_marked')  # still open at the end: marked at the last close
    return trades
