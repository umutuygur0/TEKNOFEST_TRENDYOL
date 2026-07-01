"""
25_v21b_threshold_fix.py — v21 LambdaRank + %44 threshold fix
==============================================================
Sorun: v21 %34 pozitif üretti (olması gereken ~%44)
Sebep: OOF training'i %33 pozitif → test %44 pozitif mismatch
       LambdaRank iter=1 (gruplar çok küçük, NDCG anında max)

Çözüm:
  1. BERT cache'den yükle (bert_scores_v21_train/test.npy)
  2. Features re-build (hızlı, ~3 dk)
  3. LambdaRank YENİDEN eğit ama NEG_PER_POS=4 (daha büyük gruplar)
  4. Test'te %44, %42, %46 threshold dene → 3 submission
"""

import gc, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRanker
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
CACHE      = BASE / "claude only"

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()
UNKNOWN = "unknown"

# ──── Util fonksiyonlar ────────────────────────────────────────────
TR_SUFFIXES = sorted([
    "ları","leri","lar","ler","nın","nin","nun","nün","ın","in","un","ün",
    "daki","deki","taki","teki","dan","den","tan","ten","da","de","ta","te",
    "ya","ye","yı","yi","yu","yü","la","le","ça","çe","ca","ce",
    "lik","lık","luk","lük","cı","ci","cu","cü","çı","çi","çu","çü",
    "sı","si","su","sü","sal","sel","li","lı","lu","lü","ki",
    "a","e","ı","i","u","ü",
], key=len, reverse=True)

def stem(w):
    for s in TR_SUFFIXES:
        if w.endswith(s) and len(w)-len(s) >= 3: return w[:-len(s)]
    return w

def jac(a,b):
    sa,sb = set(a.split()),set(b.split())
    return len(sa&sb)/len(sa|sb) if (sa|sb) else 0.0

def stem_jac(a,b):
    sa={stem(w) for w in a.split()}; sb={stem(w) for w in b.split()}
    return len(sa&sb)/len(sa|sb) if (sa|sb) else 0.0

def q_cov(q,t):
    qw=set(q.split())
    return len(qw&set(t.split()))/len(qw) if qw else 0.0

RENKLER = {"kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
           "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
           "bordo","haki","füme","antrasit","indigo","petrol"}
COLOR_NORM  = {"gold":"altın","silver":"gümüş","rose":"pembe","krem":"bej"}
COLOR_FAMILY= {"antrasit":"gri","füme":"gri","platin":"gri","koyu gri":"gri","açık gri":"gri",
               "lacivert":"mavi","indigo":"mavi","petrol":"mavi","saks mavi":"mavi",
               "bordo":"kırmızı","altın":"sarı","gold":"sarı","krem":"bej","ekru":"bej"}

def norm_color(c): return COLOR_NORM.get(c,c)
def color_fam(c): return COLOR_FAMILY.get(c,c)
def get_qcolor(q):
    for tok in q.split():
        if tok in RENKLER: return norm_color(tok)
    return None
def color_typed(qc,ir):
    if qc is None or not ir: return 0.0
    if qc==ir: return 2.0
    if color_fam(qc)==color_fam(ir): return 1.0
    return -1.0

def parse_attrs(s):
    if not s or s in (UNKNOWN,""): return {}
    d={}
    for part in s.split(","):
        if ":" in part:
            k,_,v=part.partition(":"); d[k.strip()]=v.strip()
    return d

MATERIAL_MAP={"pamuk":"pamuk","pamuklu":"pamuk","deri":"deri","polyester":"polyester",
              "yün":"yün","keten":"keten","çelik":"çelik"}

def mat_match(q,am):
    if not am: return 0.0
    for tok in q.split():
        m=MATERIAL_MAP.get(tok)
        if m and m in am: return 1.0
    return 0.0

def brand_tok_ovlp(q,b):
    if not b or b==UNKNOWN: return 0.0
    bt=set(t for t in re.sub(r'[^a-z0-9çğıöşü\s]',' ',b).split() if len(t)>1)
    qt=set(re.sub(r'[^a-z0-9çğıöşü\s]',' ',q).split())
    return len(bt&qt)/len(bt) if bt else 0.0

def gender_cross(q,g):
    qk=bool(re.search(r'\b(kadın|bayan)\b',q)); qe=bool(re.search(r'\berkek\b',q))
    gk=g in("kadın","bayan","kız"); ge=g=="erkek"
    if qk and ge: return -1.0
    if qe and gk: return -1.0
    if qk and gk: return  1.0
    if qe and ge: return  1.0
    return 0.0

# ──── Veri yükleme ────────────────────────────────────────────────
print("="*60, flush=True)
print("[1] VERİ YÜKLENİYOR...", flush=True)
t0=time.time()

items       = pd.read_csv(DATA/"items.csv")
terms       = pd.read_csv(DATA/"terms.csv")
train_pairs = pd.read_csv(DATA/"training_pairs.csv")
sub_pairs   = pd.read_csv(DATA/"submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col]=items[col].fillna(UNKNOWN).apply(trl)
terms["query"]=terms["query"].fillna("").apply(trl)
items["main_category"]=items["category"].str.split("/").str[0].fillna(UNKNOWN)

item_ids_arr   = items["item_id"].values
item_mains_arr = items["main_category"].values
iid_to = {row.item_id:row for row in items.itertuples()}
tid_to_q = dict(zip(terms["term_id"],terms["query"]))
print(f"  Yükleme: {time.time()-t0:.1f}s", flush=True)

# ──── Leak-free negatifler (NEG_PER_POS=4: daha büyük gruplar) ───
print("\n[2] LEAK-FREE NEGATİFLER (4/pozitif)...", flush=True)

def build_group_idx(df, cols):
    df=df.reset_index(drop=True)
    if len(cols)==1: return {k:g.index.values for k,g in df.groupby(cols[0],sort=False)}
    return {k:g.index.values for k,g in df.groupby(cols,sort=False)}

by_main   = build_group_idx(items,["main_category"])
by_gender = build_group_idx(items,["gender"])
by_mg     = build_group_idx(items,["main_category","gender"])
by_age    = build_group_idx(items,["age_group"])
by_ma     = build_group_idx(items,["main_category","age_group"])

pos_keys = set(train_pairs["term_id"].astype(str)+"\t"+train_pairs["item_id"].astype(str))
used_keys: set = set()
rng=np.random.default_rng(42)
NEG_PER_POS=4  # v21'den büyük — NDCG için daha zengin gruplar

def sample_pool(pool, tid, pos_id, max_t=40):
    if pool is None or len(pool)==0: return None
    for _ in range(max_t):
        idx=int(pool[rng.integers(0,len(pool))])
        iid=str(item_ids_arr[idx])
        k=tid+"\t"+iid
        if iid!=pos_id and k not in pos_keys and k not in used_keys: return iid
    return None

def sample_diff_main(cur_main,tid,pos_id,max_t=80):
    n=len(item_ids_arr)
    for _ in range(max_t):
        idx=int(rng.integers(0,n))
        if item_mains_arr[idx]==cur_main: continue
        iid=str(item_ids_arr[idx])
        k=tid+"\t"+iid
        if iid!=pos_id and k not in pos_keys and k not in used_keys: return iid
    return None

pos_info=train_pairs.merge(terms,on="term_id",how="left").merge(
    items[["item_id","main_category","gender","age_group"]],on="item_id",how="left")

neg_tids,neg_iids=[],[]
for row in pos_info.itertuples(index=False):
    tid=str(row.term_id); pos_id=str(row.item_id)
    main=str(row.main_category) if isinstance(row.main_category,str) else UNKNOWN
    q=str(row.query) if isinstance(row.query,str) else ""
    sel=[]

    if re.search(r'\berkek\b',q):
        pool=by_mg.get((main,"kadın"))
        if pool is None: pool=by_gender.get("kadın")
        iid=sample_pool(pool,tid,pos_id)
        if iid: sel.append(iid)
    elif re.search(r'\b(kadın|bayan)\b',q):
        pool=by_mg.get((main,"erkek"))
        if pool is None: pool=by_gender.get("erkek")
        iid=sample_pool(pool,tid,pos_id)
        if iid: sel.append(iid)

    if len(sel)<NEG_PER_POS and re.search(r'\b(bebek|çocuk)\b',q):
        pool=by_ma.get((main,"yetişkin"))
        if pool is None: pool=by_age.get("yetişkin")
        iid=sample_pool(pool,tid,pos_id)
        if iid: sel.append(iid)

    while len(sel)<NEG_PER_POS:
        iid=sample_pool(by_main.get(main),tid,pos_id)
        if iid: sel.append(iid)
        else: break

    while len(sel)<NEG_PER_POS:
        iid=sample_diff_main(main,tid,pos_id)
        if iid: sel.append(iid)
        else: break

    for iid in sel[:NEG_PER_POS]:
        k=tid+"\t"+iid
        if k in used_keys or k in pos_keys: continue
        used_keys.add(k); neg_tids.append(tid); neg_iids.append(iid)

negatives=pd.DataFrame({"term_id":neg_tids,"item_id":neg_iids,"label":0})
train_pairs["label"]=1
train_df=pd.concat([train_pairs[["term_id","item_id","label"]],
                    negatives[["term_id","item_id","label"]]],ignore_index=True)
train_df=train_df.sort_values("term_id").reset_index(drop=True)
print(f"  {len(train_df):,} | pos={train_df.label.sum():,} neg={(train_df.label==0).sum():,} | neg/pos={NEG_PER_POS}", flush=True)

# ──── BERT scores (cache'den) ──────────────────────────────────────
print("\n[3] BERT SCORES (CACHE)...", flush=True)
train_bert_path = CACHE/"bert_scores_v21_train.npy"
test_bert_path  = SUBM/"bert_scores_v21_test.npy"

if train_bert_path.exists() and len(np.load(str(train_bert_path)))==750_000:
    # v21 cache 750K ile eğitildi, biz şimdi daha fazla neg ile eğitiyoruz
    # Yeni train_df boyutu farklı → yeni BERT inference lazım
    print(f"  UYARI: train_df boyutu farklı ({len(train_df):,} vs 750K). Yeni inference yapılıyor...", flush=True)
    import torch
    from torch.amp import autocast
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset, DataLoader

    BERT_V21 = BASE/"claude only"/"models"/"bert_v21"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_V21))
    bert_mdl = AutoModelForSequenceClassification.from_pretrained(str(BERT_V21)).to(device).eval()

    def bert_infer(tids, iids, batch=256):
        scores=[]
        for i in range(0,len(tids),batch):
            bt,bi=tids[i:i+batch],iids[i:i+batch]
            qs=[tid_to_q.get(t,"") for t in bt]
            ps=[]
            for ii in bi:
                r=iid_to.get(ii)
                if r:
                    d=parse_attrs(getattr(r,"attributes","") or "")
                    renk=d.get("renk",""); mat=d.get("materyal bileşeni",d.get("materyal",""))
                    parts=[p for p in [getattr(r,"title",""),getattr(r,"brand",""),
                                       getattr(r,"category","").split("/")[0],
                                       f"renk:{renk}" if renk and renk!=UNKNOWN else "",
                                       f"mat:{mat[:25]}" if mat and mat!=UNKNOWN else ""] if p and p!=UNKNOWN]
                    ps.append(" | ".join(parts[:5]))
                else: ps.append("")
            enc=tokenizer(qs,ps,max_length=128,truncation=True,padding=True,return_tensors="pt").to(device)
            with torch.no_grad(), autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out=bert_mdl(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(out).float().cpu().tolist())
            if i%100_000==0 and i>0: print(f"    bert {i:,}/{len(tids)}", flush=True)
        return np.array(scores,dtype=np.float32)

    t1=time.time()
    new_train_bert_path=CACHE/"bert_scores_v21b_train.npy"
    if not new_train_bert_path.exists():
        print(f"  Train inference ({len(train_df):,})...", flush=True)
        train_bert=bert_infer(train_df["term_id"].tolist(),train_df["item_id"].tolist())
        np.save(str(new_train_bert_path),train_bert)
    else:
        train_bert=np.load(str(new_train_bert_path))
    print(f"  Train BERT: {(time.time()-t1)/60:.1f} dk", flush=True)
    del bert_mdl; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    train_bert=np.load(str(train_bert_path))
    print(f"  Train BERT cache ({len(train_bert):,})", flush=True)

test_bert=np.load(str(test_bert_path))
print(f"  Test BERT cache ({len(test_bert):,})", flush=True)

# ──── Feature engineering ─────────────────────────────────────────
print("\n[4] FEATURES...", flush=True)

def build_features(df_in, tfidf_v, bert_arr, fit=False, all_texts=None):
    df=df_in.merge(terms,on="term_id",how="left").merge(items,on="item_id",how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes","main_category"]:
        df[col]=df[col].fillna(UNKNOWN).apply(trl)
    if fit:
        corpus=list(df["title"])+list(df["query"])+(all_texts or [])
        tfidf_v.fit(corpus)

    qs=df["query"].tolist(); ts=df["title"].tolist()
    cats=df["category"].tolist(); brs=df["brand"].tolist()
    gens=df["gender"].tolist(); ages=df["age_group"].tolist()
    attrs=df["attributes"].tolist()

    def tccos(ql,tl,chunk=50_000):
        n=len(ql); out=np.zeros(n,dtype=np.float32)
        for i in range(0,n,chunk):
            qm=normalize(tfidf_v.transform(ql[i:i+chunk]),"l2")
            tm=normalize(tfidf_v.transform(tl[i:i+chunk]),"l2")
            out[i:i+chunk]=np.array(qm.multiply(tm).sum(axis=1)).flatten()
        return out

    parsed=[parse_attrs(a) for a in attrs]
    ar=[d.get("renk","") for d in parsed]
    am=[d.get("materyal bileşeni",d.get("materyal","")) for d in parsed]
    ak=[d.get("kol boyu",d.get("kol tipi","")) for d in parsed]
    qc=[get_qcolor(q) for q in qs]

    f=pd.DataFrame()
    f["fuzz_partial"]    =[rfuzz.partial_ratio(q,t)/100    for q,t in zip(qs,ts)]
    f["fuzz_set"]        =[rfuzz.token_set_ratio(q,t)/100  for q,t in zip(qs,ts)]
    f["fuzz_sort"]       =[rfuzz.token_sort_ratio(q,t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]      =[rfuzz.ratio(q,t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]         =[jac(q,t)                        for q,t in zip(qs,ts)]
    f["tfidf_cos"]       =tccos(qs,ts)
    f["q_cov_title"]     =[q_cov(q,t)                     for q,t in zip(qs,ts)]
    f["t_cov_query"]     =[q_cov(t,q)                     for q,t in zip(qs,ts)]
    f["cat_overlap"]     =[jac(q,c.replace("/"," "))       for q,c in zip(qs,cats)]
    f["exact_in_title"]  =[(q in t)*1.0                   for q,t in zip(qs,ts)]
    f["token_overlap"]   =[len(set(q.split())&set(t.split())) for q,t in zip(qs,ts)]
    f["age_in_q"]        =[(a not in(UNKNOWN,"") and a in q)*1.0 for a,q in zip(ages,qs)]
    f["stem_jaccard"]    =[stem_jac(q,t)                  for q,t in zip(qs,ts)]
    f["stem_cat_jac"]    =[stem_jac(q,c.replace("/"," ")) for q,c in zip(qs,cats)]
    f["gender_cross"]    =[gender_cross(q,g)              for q,g in zip(qs,gens)]
    f["first_tok_title"] =[(q.split()[0] in t if q.split() else False)*1.0 for q,t in zip(qs,ts)]
    f["first_tok_brand"] =[(q.split()[0] in b if q.split() and b!=UNKNOWN else False)*1.0 for q,b in zip(qs,brs)]
    f["q_len"]           =[len(q.split()) for q in qs]
    f["t_len"]           =[len(t.split()) for t in ts]
    f["bert_score"]      =bert_arr
    f["ana_kategori"]    =pd.Categorical([c.split("/")[0] for c in cats])
    f["color_typed"]     =[color_typed(c,r) for c,r in zip(qc,ar)]
    f["has_q_color"]     =[(c is not None)*1.0 for c in qc]
    f["brand_tok_ovlp"]  =[brand_tok_ovlp(q,b) for q,b in zip(qs,brs)]
    f["material_match"]  =[mat_match(q,a) for q,a in zip(qs,am)]
    f["attr_jac_fixed"]  =[jac(q," ".join(d.values())) if (d:=parse_attrs(a)) else 0.0 for q,a in zip(qs,attrs)]
    f["attr_q_cov"]      =[q_cov(q," ".join(d.values())) if (d:=parse_attrs(a)) else 0.0 for q,a in zip(qs,attrs)]
    return f, df

t1=time.time()
tfidf=TfidfVectorizer(ngram_range=(1,2),max_features=60_000,sublinear_tf=True,min_df=2)
FEATS=["fuzz_partial","fuzz_set","fuzz_sort","fuzz_basic","jaccard","tfidf_cos",
       "q_cov_title","t_cov_query","cat_overlap","exact_in_title","token_overlap","age_in_q",
       "stem_jaccard","stem_cat_jac","gender_cross","first_tok_title","first_tok_brand",
       "q_len","t_len","bert_score","ana_kategori",
       "color_typed","has_q_color","brand_tok_ovlp","material_match","attr_jac_fixed","attr_q_cov"]

X_tr,df_tr=build_features(train_df,tfidf,train_bert,fit=True,
                           all_texts=items["title"].tolist()+terms["query"].tolist())
y=train_df["label"].values
tids_arr=train_df["term_id"].values
print(f"  {X_tr[FEATS].shape} | {time.time()-t1:.1f}s", flush=True)

# ──── LambdaRank 5-fold ────────────────────────────────────────────
print("\n[5] LAMBDARANK 5-FOLD...", flush=True)
oof=np.zeros(len(train_df),dtype=np.float32)
models=[]
gkf=GroupKFold(n_splits=5)

for fold,(tri,vli) in enumerate(gkf.split(X_tr,y,tids_arr)):
    tr_df_idx=pd.DataFrame({"tid":tids_arr[tri],"pos":tri}).sort_values("tid")
    vl_df_idx=pd.DataFrame({"tid":tids_arr[vli],"pos":vli}).sort_values("tid")
    tri_s=tr_df_idx["pos"].values; vli_s=vl_df_idx["pos"].values

    _,tr_grp=np.unique(tids_arr[tri_s],return_counts=True)
    _,vl_grp=np.unique(tids_arr[vli_s],return_counts=True)

    model=LGBMRanker(
        objective="lambdarank", n_estimators=1000, learning_rate=0.05,
        num_leaves=127, max_depth=8, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42+fold, n_jobs=-1, verbose=-1, label_gain=[0,1]
    )
    model.fit(X_tr[FEATS].iloc[tri_s], y[tri_s], group=tr_grp,
              eval_set=[(X_tr[FEATS].iloc[vli_s],y[vli_s])],
              eval_group=[vl_grp], eval_at=[5,10],
              callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(0)])
    val_sc=model.predict(X_tr[FEATS].iloc[vli_s])
    oof[vli_s]=val_sc.astype(np.float32)
    models.append(model)
    bf=0.0
    for q in np.arange(0.4,0.75,0.05):
        t=np.quantile(val_sc,q)
        s=f1_score(y[vli_s],(val_sc>t).astype(int),average="macro")
        if s>bf: bf=s
    print(f"  Fold {fold+1} | iter={model.best_iteration_} | F1≈{bf:.4f}", flush=True)

best_thr,best_f1=0.0,0.0
for q in np.arange(0.40,0.75,0.01):
    t=np.quantile(oof,q)
    s=f1_score(y,(oof>t).astype(int),average="macro")
    if s>best_f1: best_f1,best_thr=s,t
print(f"\n  OOF F1={best_f1:.5f} | threshold={best_thr:.4f}", flush=True)

fi=pd.Series(sum(m.feature_importances_ for m in models)/len(models),
             index=FEATS).sort_values(ascending=False)
print("\n  Feature Önemleri (top 15):", flush=True)
for feat,imp in fi.head(15).items():
    print(f"    {feat:25s}: {imp:5.0f}", flush=True)

# ──── Submission: 3 threshold dene ────────────────────────────────
print("\n[6] SUBMISSION (3 threshold)...", flush=True)
X_te,_=build_features(sub_pairs,tfidf,test_bert)
test_sc=sum(m.predict(X_te[FEATS]) for m in models)/len(models)

# Test skoru kaydet
np.save(str(SUBM/"v21b_test_lgbm_scores.npy"), test_sc.astype(np.float32))

for target_pct in [42, 44, 46]:
    q_thr=1.0 - target_pct/100.0
    thr=np.quantile(test_sc,q_thr)
    pred=(test_sc>thr).astype(int)
    pos_n=pred.sum()
    fname=f"submission_v21b_{target_pct}pct.csv"
    pd.DataFrame({"id":sub_pairs["id"],"prediction":pred}).to_csv(str(SUBM/fname),index=False)
    print(f"  {target_pct}%: pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%) → {fname}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v21b", flush=True)
print(f"  OOF F1    : {best_f1:.5f}", flush=True)
print(f"  Süre      : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"{'='*60}", flush=True)
