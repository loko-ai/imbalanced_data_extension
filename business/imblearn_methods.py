from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from imblearn.under_sampling import RandomUnderSampler




# def
from loguru import logger

smote_params = ["sampling_strategy", "k_neighbors"]
under_params = ["sampling_strategy", "replacement"]
over_params = ["sampling_strategy", "shrinkage"]

#oversampling strategy: {'not majority', 'not minority', 'auto', 'all', 'minority'}

class Resampler:
    def __init__(self, method, sampling_strategy="auto", random_state=None, **kwargs):
        self.method = method
        self.sampler = None
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state if random_state else 144
        self.__dict__.update(kwargs)

    def _oversampler_definition(self):
        ros = RandomOverSampler(random_state=self.random_state, sampling_strategy=self.sampling_strategy)#, shrinkage=self.shrinkage
        return ros

    def _undersampling_definition(self):
        rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=self.sampling_strategy, replacement=self.replacement)
        return rus

    def _smote_definition(self):
        logger.debug(f"kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk {self.k_neighbors}")
        sm = SMOTE(random_state=self.random_state, sampling_strategy=self.sampling_strategy, k_neighbors = self.k_neighbors)
        return sm

    def _smoten_definition(self):
        sm = SMOTEN(random_state=self.random_state, sampling_strategy=self.sampling_strategy, k_neighbors = self.k_neighbors)
        return sm

    def __call__(self, X, y, *args, **kwargs):
        if self.method=="undersampling":
            self.sampler = self._undersampling_definition()
        elif self.method=="oversampling":
            self.sampler = self._oversampler_definition()
        elif self.method=="SMOTE":
            self.sampler = self._smote_definition()
        elif self.method=="SMOTEN":
            self.sampler = self._smoten_definition()
        else:
            raise Exception(f"No sampling method specified or method not yet supported... Method name: {self.method}")
        from collections import Counter

        X_sampled, y_sampled = self.sampler.fit_resample(X, y)
        return X_sampled, y_sampled


if __name__ == '__main__':
    from collections import Counter
    from sklearn.datasets import make_classification
    X, y = make_classification(n_classes=2, class_sep=2,
     weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    n_features=5, n_clusters_per_class=1, n_samples=60, random_state=10)
    res = dict(X=X.tolist(), y=y.tolist())
    print(res)
    print("before: ",Counter(y))

    resampler = Resampler(method="undersampling", sampling_strategy="auto", random_state=144, replacement=False)

    X_sampled, y_sampled = resampler(X, y)
    print("after under sampling: ",Counter(y_sampled))

    resampler = Resampler(method="oversampling", sampling_strategy="all", random_state=144, shrinkage=None)

    X_sampled, y_sampled = resampler(X, y)
    print("after oversampling: ",Counter(y_sampled))

    resampler = Resampler(method="SMOTE", sampling_strategy="auto", random_state=144, k_neighbors = 4)

    X_sampled, y_sampled = resampler(X, y)
    print("after smote: ",Counter(y_sampled))
