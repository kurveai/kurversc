"""Research-only nonnegative mixture median; no default or graph changes."""

import numpy as np


LEVELS = np.array([.05,.25,.5,.75,.95])


def mixture_median(probability_positive, positive_log_quantiles):
    """Median of a point mass at zero plus a conditional positive CDF."""
    probability=np.asarray(probability_positive,dtype=float).reshape(-1)
    quantiles=np.asarray(positive_log_quantiles,dtype=float)
    if (quantiles.shape!=(len(probability),len(LEVELS)) or not np.isfinite(quantiles).all()
            or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1))):
        raise ValueError("Aligned finite conditional probabilities/quantiles required")
    quantiles=np.maximum.accumulate(np.maximum(quantiles,0),axis=1)
    conditional=np.maximum(0,1-.5/np.maximum(probability,1e-12))
    result=np.zeros(len(probability))
    for row in np.flatnonzero(probability>.5):
        result[row]=np.expm1(np.interp(conditional[row],np.r_[0.,LEVELS],np.r_[0.,quantiles[row]]))
    return result


class ZeroAwareQuantileReadout:
    def __init__(self,*,seed=0,model_params=None):
        self.seed,self.params=seed,dict(model_params or {})

    def fit(self,features,labels,*,categorical):
        from catboost import CatBoostClassifier,CatBoostRegressor

        y=np.asarray(labels,dtype=float)
        if (len(features)!=len(y) or not np.isfinite(y).all() or np.any(y<0)
                or np.sum(y>0)<20 or np.sum(y==0)<20):
            raise ValueError("Nonnegative numeric targets with >=20 positive and zero observations required")
        allowed={"iterations","depth","learning_rate","l2_leaf_reg","thread_count"}
        if set(self.params)-allowed:
            raise ValueError("Unsupported research learner parameters")
        params=dict(iterations=300,depth=6,learning_rate=.05,l2_leaf_reg=10.,thread_count=4,
                    random_seed=self.seed,verbose=False,allow_writing_files=False)
        params.update(self.params)
        self.occurrence=CatBoostClassifier(loss_function="Logloss",**params).fit(features,(y>0).astype(int),cat_features=categorical)
        self.positive=CatBoostRegressor(loss_function="MultiQuantile:alpha="+",".join(map(str,LEVELS)),**params).fit(
            features.loc[y>0],np.log1p(y[y>0]),cat_features=categorical)
        self.audit=dict(strategy="zero_mass_plus_conditional_positive_quantiles",rows=len(y),positive_rows=int((y>0).sum()),
            zero_share=float((y==0).mean()),quantiles=LEVELS.tolist(),target_transform="log1p, training-only",
            trees_per_component=params["iterations"],components=2,
            compute_note="Two supervised heads; not claimed equal compute to one direct-MAE head")
        return self

    def predict(self,features):
        return mixture_median(self.occurrence.predict_proba(features)[:,1],self.positive.predict(features))
