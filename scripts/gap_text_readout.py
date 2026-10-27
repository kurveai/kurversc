"""Experimental field-aware sparse text head; no KurveRSC integration."""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class FieldTextReadout:
    """Keep the column/role of words, with an explicit vocabulary ceiling."""
    def __init__(self, *, regularization=1., maximum_per_field=4096):
        self.regularization,self.maximum_per_field=regularization,maximum_per_field

    @staticmethod
    def documents(frame,name):
        return frame[name].astype("string").fillna("").replace("__missing__","").str.slice(0,2048).tolist()

    def fit(self,frame,labels,columns):
        if not columns or len(columns)>8 or len(labels)!=len(frame):
            raise ValueError("Aligned training examples and at most eight text fields required")
        self.vectorizers={}
        matrices=[]
        for name in columns:
            vectorizer=TfidfVectorizer(max_features=self.maximum_per_field,ngram_range=(1,2),
                min_df=2,max_df=.98,sublinear_tf=True,strip_accents="unicode",dtype=np.float32)
            try:
                matrix=vectorizer.fit_transform(self.documents(frame,name))
            except ValueError as exc:
                if "empty vocabulary" not in str(exc) and "After pruning" not in str(exc):
                    raise
                continue
            self.vectorizers[name]=vectorizer
            matrices.append(matrix)
        if not matrices:
            raise ValueError("No field has a reusable training vocabulary")
        self.model=LogisticRegression(C=self.regularization,solver="liblinear",max_iter=1000,random_state=0)
        self.model.fit(sparse.hstack(matrices,format="csr"),labels.astype(int))
        if not np.array_equal(self.model.classes_,np.array([0,1])):
            raise ValueError("Ordinary binary targets required")
        self.audit=dict(training_rows=len(frame),regularization=self.regularization,
            vocabulary_sizes={name:len(vectorizer.vocabulary_) for name,vectorizer in self.vectorizers.items()},
            maximum_per_field=self.maximum_per_field,maximum_fields=8,
            scope="Each field vocabulary and classifier fitted on mature training support only")
        return self

    def predict_proba(self,frame):
        matrices=[vectorizer.transform(self.documents(frame,name)) for name,vectorizer in self.vectorizers.items()]
        return self.model.predict_proba(sparse.hstack(matrices,format="csr"))
