# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpy import mean, std
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from preprocessing import train_copy
from embedding import wordvec_df


# %%
import sys
sys.path.insert(1, '/home/arthur/TCC/codigo')


# %%
model = GaussianProcessClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# %%
# ext_scores = cross_val_score(model, wordvec_df, train_copy.cEXT,
#                              scoring='accuracy', cv=cv, n_jobs=-1, pre_dispatch=1)


# %%
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1 *
                  RationalQuadratic(), 1*WhiteKernel()]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(wordvec_df, train_copy.cEXT)

# %%
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# %%
# print('Mean Accuracy: %.3f (%.3f)' % (mean(ext_scores), std(ext_scores)))


# %%
