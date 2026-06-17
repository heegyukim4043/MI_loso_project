"""Extra cross-dataset CSP-LDA variants beyond EA ablations.

Runs on raw-unified Cho2017/Lee2019 and optional PhysioNet pivot.
Variants:
- gaussian_ot_source_to_target: Bures/Gaussian OT channel covariance map on source, then CSP-LDA.
- gaussian_ot_then_subject_ea: Gaussian OT source->target followed by subject EA.
- frequency_specific_ea: mu/beta band-specific DatasetEA+SubjectEA, CSP features concatenated.
- physionet_pivot_dataset: color both source and target to PhysioNet dataset covariance reference.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from cross_dataset import STANDARD_MI_CHANNELS, RESULTS_DIR
from eeg_ea import euclidean_align, apply_ea_loso
from mrfbcsp_loso import ManualCSP

RAW_UNIFIED=(ROOT/'..'/'preprocessed_raw_unified').resolve()
PREPROCESSED=(ROOT/'..'/'preprocessed').resolve()
SFREQ=128.0


def load(name, raw=True):
    base=RAW_UNIFIED if raw and name in {'cho2017','lee2019'} else PREPROCESSED
    d=np.load(base/f'{name}.npz', allow_pickle=True)
    return d['X'].astype(np.float32), d['y'].astype(np.int64), d['subjects'].astype(np.int64), [str(c) for c in d['ch_names']]


def select_common(X, ch, common):
    return X[:, [ch.index(c) for c in common], :]


def load_pair(train_name,test_name, include_physionet=False):
    Xtr,ytr,strn,chtr=load(train_name)
    Xte,yte,ste,chte=load(test_name)
    common=[c for c in STANDARD_MI_CHANNELS if c in chtr and c in chte]
    Xp=yp=sp=chp=None
    if include_physionet:
        Xp,yp,sp,chp=load('physionet', raw=False)
        common=[c for c in common if c in chp]
        Xp=select_common(Xp,chp,common)
    return select_common(Xtr,chtr,common),ytr,strn,select_common(Xte,chte,common),yte,ste,common,Xp


def cov_mean(X, eps=1e-6):
    C=np.zeros((X.shape[1],X.shape[1]),dtype=np.float64)
    for x in X.astype(np.float64):
        z=x-x.mean(axis=1,keepdims=True)
        c=z@z.T/max(z.shape[1]-1,1)
        tr=np.trace(c)
        if tr>1e-12:
            c=c/tr
        C+=c
    C/=len(X)
    C=(C+C.T)/2+eps*np.eye(C.shape[0])
    return C


def invsqrt(C,eps=1e-8):
    vals,vecs=np.linalg.eigh((C+C.T)/2)
    vals=np.maximum(vals,eps)
    return vecs@np.diag(vals**-0.5)@vecs.T


def matsqrt(C):
    vals,vecs=np.linalg.eigh((C+C.T)/2)
    vals=np.maximum(vals,1e-8)
    return vecs@np.diag(np.sqrt(vals))@vecs.T


def apply_A(X,A):
    return np.einsum('cd,ndt->nct',A,X).astype(np.float32)


def zscore_train_test(Xtr,Xte):
    mu=Xtr.mean(axis=(0,2),keepdims=True); sd=Xtr.std(axis=(0,2),keepdims=True)+1e-8
    return ((Xtr-mu)/sd).astype(np.float32), ((Xte-mu)/sd).astype(np.float32)


def gaussian_ot_map(Cs,Ct):
    # Map source covariance Cs to target covariance Ct for zero-mean Gaussians.
    Cs_sqrt=matsqrt(Cs); Cs_inv=invsqrt(Cs)
    middle=matsqrt(Cs_sqrt@Ct@Cs_sqrt)
    A=Cs_inv@middle@Cs_inv
    return np.real(A).astype(np.float64)


def band_filter(X, lo, hi):
    sos=butter(4, [lo/(SFREQ/2), hi/(SFREQ/2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, X, axis=2).astype(np.float32)


def csp_features(Xtr,ytr,Xte,n_csp=8):
    csp=ManualCSP(n_components=n_csp)
    csp.fit(Xtr,ytr)
    return csp.transform(Xtr), csp.transform(Xte)


def fit_predict_csp_lda(Xtr,ytr,Xte,n_csp=8):
    Ftr,Fte=csp_features(Xtr,ytr,Xte,n_csp=n_csp)
    sc=StandardScaler(); Ftr=sc.fit_transform(Ftr); Fte=sc.transform(Fte)
    lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    lda.fit(Ftr,ytr)
    return lda.predict(Fte)


def fit_predict_multiband(Xtr,ytr,Xte,bands=((8,12),(13,30)),n_csp=6):
    feats_tr=[]; feats_te=[]
    for lo,hi in bands:
        Bt=band_filter(Xtr,lo,hi); Be=band_filter(Xte,lo,hi)
        Ftr,Fte=csp_features(Bt,ytr,Be,n_csp=n_csp)
        feats_tr.append(Ftr); feats_te.append(Fte)
    Ftr=np.concatenate(feats_tr,axis=1); Fte=np.concatenate(feats_te,axis=1)
    sc=StandardScaler(); Ftr=sc.fit_transform(Ftr); Fte=sc.transform(Fte)
    lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    lda.fit(Ftr,ytr)
    return lda.predict(Fte)


def apply_variant(variant,Xtr,ytr,strn,Xte,ste,Xp=None):
    if variant=='gaussian_ot_source_to_target':
        A=gaussian_ot_map(cov_mean(Xtr),cov_mean(Xte)); Xtr=apply_A(Xtr,A); Xtr,Xte=zscore_train_test(Xtr,Xte)
        return Xtr,Xte,'single'
    if variant=='gaussian_ot_then_subject_ea':
        A=gaussian_ot_map(cov_mean(Xtr),cov_mean(Xte)); Xtr=apply_A(Xtr,A)
        Xtr=apply_ea_loso(Xtr,strn); Xte=apply_ea_loso(Xte,ste); Xtr,Xte=zscore_train_test(Xtr,Xte)
        return Xtr,Xte,'single'
    if variant=='frequency_specific_ea':
        Xtr=euclidean_align(Xtr); Xte=euclidean_align(Xte)
        Xtr=apply_ea_loso(Xtr,strn); Xte=apply_ea_loso(Xte,ste); Xtr,Xte=zscore_train_test(Xtr,Xte)
        return Xtr,Xte,'multiband'
    if variant=='physionet_pivot_dataset':
        if Xp is None: raise ValueError('PhysioNet pivot required')
        Rp=cov_mean(Xp); Ap=matsqrt(Rp)
        Xtr=apply_A(Xtr, Ap@invsqrt(cov_mean(Xtr)))
        Xte=apply_A(Xte, Ap@invsqrt(cov_mean(Xte)))
        Xtr,Xte=zscore_train_test(Xtr,Xte)
        return Xtr,Xte,'single'
    raise ValueError(variant)


def run_one(src,tgt,variant,run_id):
    include_p=variant=='physionet_pivot_dataset'
    Xtr,ytr,strn,Xte,yte,ste,common,Xp=load_pair(src,tgt,include_physionet=include_p)
    print('='*70); print(f'{variant}: {src}->{tgt} Xtr={Xtr.shape} Xte={Xte.shape} ch={len(common)}')
    Xtr,Xte,mode=apply_variant(variant,Xtr,ytr,strn,Xte,ste,Xp)
    out=Path(RESULTS_DIR)/f'loso_results_{run_id}_{variant}_cross_{src}_to_{tgt}_csp_lda.csv'
    fields=['train_ds','test_ds','variant','subject','n_test','acc','bac','kappa']
    with out.open('w',newline='') as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    rows=[]
    for s in np.unique(ste):
        m=ste==s
        pred=fit_predict_multiband(Xtr,ytr,Xte[m]) if mode=='multiband' else fit_predict_csp_lda(Xtr,ytr,Xte[m])
        yt=yte[m]
        row={'train_ds':src,'test_ds':tgt,'variant':variant,'subject':int(s),'n_test':int(m.sum()),'acc':round(accuracy_score(yt,pred)*100,2),'bac':round(balanced_accuracy_score(yt,pred)*100,2),'kappa':round(cohen_kappa_score(yt,pred),3)}
        rows.append(row)
        with out.open('a',newline='') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
    acc=float(np.mean([r['acc'] for r in rows])); kap=float(np.mean([r['kappa'] for r in rows]))
    print(f'{src}->{tgt}: {acc:.2f}% k={kap:.3f} saved={out}')
    return acc,kap,out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants',nargs='+',default=['gaussian_ot_source_to_target','gaussian_ot_then_subject_ea','frequency_specific_ea','physionet_pivot_dataset'])
    ap.add_argument('--run_id',default=None)
    args=ap.parse_args()
    run_id=args.run_id or datetime.now().strftime('%Y%m%d_extra_variants')
    summary=[]
    for v in args.variants:
        r1=run_one('cho2017','lee2019',v,run_id); r2=run_one('lee2019','cho2017',v,run_id)
        summary.append((v,r1,r2))
    out=Path(RESULTS_DIR)/f'cross_dataset_extra_variants_summary_{run_id}.csv'
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['variant','cho_to_lee_acc','cho_to_lee_kappa','lee_to_cho_acc','lee_to_cho_kappa','cho_to_lee_csv','lee_to_cho_csv'])
        for v,r1,r2 in summary:
            w.writerow([v,f'{r1[0]:.2f}',f'{r1[1]:.3f}',f'{r2[0]:.2f}',f'{r2[1]:.3f}',r1[2],r2[2]])
    print(f'summary={out}')
    prog=ROOT.parent/'progress.md'
    lines=['',f'## Cross-Dataset Extra Variant Sweep ({datetime.now():%Y-%m-%d %H:%M})','', '- Status: `completed`', f'- Summary CSV: `{out}`','', '| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |','|---|---:|---:|']
    for v,r1,r2 in summary:
        lines.append(f'| {v} | {r1[0]:.2f}% / k={r1[1]:.3f} | {r2[0]:.2f}% / k={r2[1]:.3f} |')
    prog.write_text(prog.read_text()+'\n'.join(lines)+'\n')

if __name__=='__main__': main()
