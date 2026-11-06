import numpy as np
SR=22050; x=np.memmap('gt3/mono22k.f32',dtype=np.float32,mode='r')
hop=1102; win=4096; nfft=16384; w=np.hanning(win)
n=(len(x)-win)//hop
df=SR/nfft
R=np.arange(900,9600,10.0)            # rpm candidates
K=np.arange(1,25)                       # half-orders 0.5..12
on=np.round(np.outer(R/120,K)/df).astype(int)
off=np.round(np.outer(R/120,K-0.5)/df).astype(int)
wk=np.where(K%6==0,1.5,1.0)             # orders 3,6,9,12 slightly favoured
rpm=np.zeros(n,np.float32); sc=np.zeros(n,np.float32); db=np.zeros(n,np.float32); hi=np.zeros(n,np.float32)
B=2048
for s0 in range(0,n,B):
    m=min(B,n-s0)
    idx=(np.arange(m)[:,None]+s0)*hop+np.arange(win)[None,:]
    fr=np.asarray(x[s0*hop:(s0+m-1)*hop+win])
    seg=fr[(np.arange(m)[:,None])*hop+np.arange(win)[None,:]]
    db[s0:s0+m]=10*np.log10((seg**2).mean(1)+1e-12)
    Sp=np.abs(np.fft.rfft(seg*w,nfft,axis=1))
    hi[s0:s0+m]=10*np.log10((Sp[:,int(2000/df):int(8000/df)]**2).sum(1)/(Sp**2).sum(1)+1e-12)
    L=np.log(Sp+1e-6)
    L-=np.median(L[:,:int(1200/df)],axis=1,keepdims=True)
    son=(L[:,on]*wk).sum(2)/wk.sum(); soff=L[:,off].mean(2)
    s=son-soff
    j=s.argmax(1); rpm[s0:s0+m]=R[j]; sc[s0:s0+m]=s[np.arange(m),j]
np.savez('gt3/rpm.npz',rpm=rpm,sc=sc,db=db,hi=hi,hop=hop/SR)
print(n, np.percentile(sc,[10,50,90]))
