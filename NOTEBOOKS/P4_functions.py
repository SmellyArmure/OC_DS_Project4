import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import seaborn as sns

from sklearn.decomposition import pca
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn import metrics

def move_cat_containing(my_index, strings, order):
  idx_sel = []
  if order == 'last':
    index = my_index
  elif order == 'first':
    index = my_index[::-1]
  else:
    print("--- WARNING : index unchanged.\n -- Wrong order passed. Pass 'first' or 'last'")
    return my_index
  for s in strings:
    idx_sel += [i for i,x in enumerate(index) if x in index[index.str.contains(s)]]
  to_move = index[idx_sel]
  rank = max(idx_sel)
  mod_index = index.drop(to_move)
  for c in to_move:
    mod_index = mod_index.insert(rank,c)
  return mod_index if order=='last' else mod_index[::-1]

# Printing total nb and percentage of null

def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
        '{:.1f}'.format(tot_null*100/(df.shape[0]*df.shape[1])))

# Displaying number of missing values per column

def plot_export_missing(df, cols, n_file, title,
                        shorten_label=False, figsize=(12,8), save_enabled=False):
    with plt.style.context('default'):
        fig, axs = plt.subplots(2,1)
        msno.matrix(df[cols] , sparkline=False,
                    fontsize=11, ax=axs[0])
        msno.bar(df[cols], ax=axs[1], fontsize=11)
        if shorten_label:
            for ax in axs:
                lab = [item.get_text() for item in ax.get_xticklabels()]
                short_lab = [s[:7]+'...'+s[-7:] if len(s)>14 else s for s in lab]
                ax.axes.set_xticklabels(short_lab)
    fig.set_size_inches(figsize)
    [ax.grid() for ax in axs.flatten()];
    [sns.despine(ax=ax, right=False, left=False,top=False, bottom=False)\
                                        for ax in axs.flatten()];
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.suptitle(title, fontweight='bold', fontsize=14)
    if not os.path.exists(os.getcwd()+'/FIG'):
        os.makedirs('FIG')
    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+n_file, dpi=400);
    plt.show()

# Plotting histograms of specified quantitative continuous columns of a dataframe and mean, median and mode values.

def plot_histograms(df, cols, file_name=None, figsize=(12,7), layout=(3,3), save_enabled=False):

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        ax.hist(df[c],  bins=30, color='grey')
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        
    plt.tight_layout(w_pad=0.5, h_pad=0.65)

    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+file_name,
                                dpi=400);
    plt.show()

# normality tests

from scipy.stats import shapiro, normaltest, anderson

def normality_tests(data, print_opt=False):
    res_df = pd.DataFrame([])
    data_notna = data[data.notna()]
    # Shapiro-Wilk - D'Agostino's K^2
    for f_name, func in zip(['Shapiro-Wilk', "D'Agostino K^2"],[shapiro, normaltest]):
        stat, p = func(data_notna)
        res_df.loc[f_name,'stat'] = stat
        res_df.loc[f_name,'p_value'] = p
        if print_opt: print('---'+f_name) 
        if print_opt: print('stat=%.3f, p=%.3f' % (stat, p))
        res_df.loc[f_name,'res'] = [p > 0.05]
        if p > 0.05:
            if print_opt: print('Probably Gaussian')
        else:
            if print_opt: print('Probably not Gaussian')
    # Anderson-Darling
    result = anderson(data_notna)
    if print_opt: print('---'+'Anderson-Darling')
    res_df.loc['Anderson-Darling','stat'] = result.statistic
    if print_opt: print('stat=%.3f' % (result.statistic))
    res_and = [(int(result.significance_level[i]),result.statistic < res)\
                   for i,res in enumerate(result.critical_values)]
    res_df.loc['Anderson-Darling','res'] = str(res_and)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            if print_opt: print('Probably Gaussian at the %.1f%% level' % (sl))
        else:
            if print_opt: print('Probably not Gaussian at the %.1f%% level' % (sl))
    return res_df


# plotting histograms, qq plots and printing the results of normality tests 
from statsmodels.graphics.gofplots import qqplot

def plot_hist_qqplot(data, name, save=False):
    fig, axs = plt.subplots(1,2)
    # histogram
    axs[0].hist(data, histtype='stepfilled',ec='k', color='lightgrey', bins =25);
    # using statsmodels qqplot's module
    qqplot(data, line='r', **{'markersize': 5, 'mec': 'k','color': 'lightgrey'}, ax=axs[1])
    plt.gcf().set_size_inches(10,2.5)
    fig.suptitle(name, fontweight='bold', size=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

# Plotting bar plots of the main categorical columns

def plot_barplots(df, cols, file_name=None, figsize=(12,7), layout=(2,3), save_enabled=False):

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat>15:
            ser[0:15].plot.bar(color='grey',ec='k', ax=ax)
        else:
            ser.plot.bar(color='grey',ec='k',ax=ax)
        ax.set_title(c[0:17]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s)>7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400);
    plt.show()


# Testing for linear correlation (Pearson) and for monotonic relationship (Spearman, Kendall).

from scipy.stats import pearsonr, spearmanr, kendalltau

def correlation_tests(data1, data2, print_opt=False):
    res_df = pd.DataFrame([])
    # data1_notna = data1[data1.notna()]
    # Pearson, Spearman, Kendall
    for f_name, func in zip(['Pearson', 'Spearman', 'Kendall'],[pearsonr, spearmanr, kendalltau]):
        stat, p = func(data1, data2)
        res_df.loc[f_name,'stat'] = stat
        res_df.loc[f_name,'p_value'] = p
        if print_opt: print('---'+f_name)
        if print_opt: print('stat=%.3f, p=%.3f' % (stat, p))
        if print_opt: print('Probably independent') if p > 0.05 else print('Probably dependent')
    return res_df

################### KNN Imputation ########################

def naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr):
    # Model
    print('--- model: {:.3}'.format(metrics.r2_score(y_te, y_pr)))
    # normal random distribution
    y_pr_rand = np.random.normal(0,1, y_pr.shape)
    print('--- normal random distribution: {:.3}'\
          .format(metrics.r2_score(y_te, y_pr_rand)))
    # dummy regressors
    for s in ['mean', 'median']:
        dum = DummyRegressor(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy regressor ('+ s +') : r2_score={:.3}'\
              .format(metrics.r2_score(y_te, y_pr_dum)))

def naive_model_compare_acc_f1(X_tr, y_tr, X_te, y_te, y_pr, average='weighted'):
    def f1_prec_recall(yte, ypr):
        prec = metrics.precision_score(yte, ypr, average=average)
        rec = metrics.recall_score(yte, ypr, average=average)
        f1 = metrics.f1_score(yte, ypr, average=average)
        return [f1, prec, rec]
    # Model
    print('--- model: f1={:.3}, precision={:.3}, recall={:.3}'\
                                             .format(*f1_prec_recall(y_te, y_pr)))
    # Dummy classifier
    for s in ['stratified','most_frequent','uniform']:
        dum = DummyClassifier(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy class. ('+ s\
              +'): f1={:.3}, precision={:.3}, recall={:.3}'\
                                             .format(*f1_prec_recall(y_te, y_pr_dum)))

def plot_hist_pred_val(y_te, y_pr, y_pr_, bins=150, xlim=(0,20), short_lab=False):
    # Plotting dispersion of data to be imputed
    bins = plt.hist(y_te, alpha=0.5, color='b', bins=bins, density=True,
                    histtype='step', lw=3, label='y_te (real val. from test set)')[1]
    ax=plt.gca()
    ax.hist(y_pr, alpha=0.5, color='g', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr (pred. val. from test set)');
    ax.hist(y_pr_, alpha=0.5, color='r', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr_ (pred. val. to be imputed)');
    ax.set(xlim=xlim)
    plt.xticks(rotation=45, ha='right')
    plt.draw()
    if short_lab:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s)>7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
    ax.legend(loc=1)
    plt.title("Frequency of values", fontweight='bold', fontsize=12)
    plt.gcf().set_size_inches(6,2)
    plt.show()

# Data Preprocessing for quantitative and categorical data with encoding options
def data_preprocessing(df, var_model, var_target, enc_strat_cat='label'):
    ## Data Processing
    df_train = df[var_model+[var_target]].copy('deep')
    if df[var_model].isna().sum().sum()!=0 :
        print("ERROR preprocessing: var_model columns should not contain nan !!!")
        return None, None
    else:
        cat_cols = df_train[var_model].select_dtypes('object').columns
        num_cols = df_train[var_model].select_dtypes(include=np.number).columns
        # # Encoding categorical values
        if enc_strat_cat == 'label':
        # --- OPTION 1: Label Encoding categorical values
            for c in cat_cols:
                df_train[c] = LabelEncoder().fit_transform(df_train[c].values)
        elif enc_strat_cat == 'hashing':
        # --- OPTION 2: Feature hashing of categorical values
            for c in cat_cols:
                df_train[c] = df_train[c].astype('str')
                n_feat = 5
                hasher = FeatureHasher(n_features=n_feat, input_type='string')
                f = hasher.transform(df_train[c])
                arr = pd.DataFrame(f.toarray(), index=df_train.index)
                df_train[[c+'_'+str(i+1) for i in range(n_feat)]] = pd.DataFrame(arr)
                del df_train[c]
                cols = list(df_train.columns)
                cols.remove(var_target)
                df_train = df_train.reindex(columns=cols+[var_target])
        else:
            print("ERROR: Wrong value of enc_strat_cat")
            return None, None
        # # Standardizing quantitative values
        if len(list(num_cols)):
            df_train[num_cols] = \
                      StandardScaler().fit_transform(df_train[num_cols].values)
        # Splitting in X and y, then in training and testing set
        X = df_train.iloc[:,:-1].values
        y = df_train.iloc[:,-1].values
        return X, y

# Data preprocessing, Knn training, then predicting all-in-one
# inferring 'pnns1' (2_C) from .... + 'pnns2' (2_C)
# Works for both quantitative (knnregressor)
# and categorical (knnclassifier) target features

def knn_impute(df, var_model, var_target, enc_strat_cat='label',
                   clip=None, plot=True):
    
    if df[var_target].isna().sum()==0:
        print('ERROR: Nothing to impute (target column already filled)')
        return None, None
    else :
        if df[var_target].dtype =='object':
            # knn classifier
            print('ooo----KNN CLASSIFICATION :',var_target)
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            gsCV = GridSearchCV(KNeighborsClassifier(),
                            {'n_neighbors': [5,7,9,11,13]},
                            cv=skf, return_train_score=True,
                            scoring='f1_weighted')
            mod = 'class'
        elif df[var_target].dtype in ['float64', 'int64']:
            # knn regressor
            print('ooo----KNN REGRESSION :',var_target)
            kf = KFold(n_splits=5, shuffle=True)
            gsCV = GridSearchCV(KNeighborsRegressor(),
                            {'n_neighbors': [3,5,7,9,11,13]},
                            cv=kf, return_train_score=True)
            mod = 'reg'
        else:
            print("ERROR: dtype of target feature unknown")
        ## Data Preprocessing
        X, y = data_preprocessing(df.dropna(subset=var_model+[var_target]),
                                var_model=var_model, var_target=var_target,
                                enc_strat_cat=enc_strat_cat)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        ## Training KNN
        gsCV.fit(X_tr, y_tr)
        res = gsCV.cv_results_
        ## Predicting test set with the model and clipping
        y_pr = gsCV.predict(X_te)
        try:
            if clip: y_pr = y_pr.clip(*clip) # regressor option only
        except:
            print("ERROR: clip available for regressor option only") 
        # Comparison with naive baselines
        if mod == 'class':
            naive_model_compare_acc_f1(X_tr,y_tr,X_te,y_te,y_pr,average='micro')
        elif mod == 'reg':
            naive_model_compare_r2(X_tr,y_tr,X_te,y_te,y_pr)
        else:
            print("ERROR: check type of target feature...")
        ## Predicting using knn
        ind_to_impute = df.loc[df[var_target].isna()].index 
        X_, y_ = data_preprocessing(df.loc[ind_to_impute], var_model=var_model,
                                    var_target=var_target,
                                    enc_strat_cat=enc_strat_cat)
        # Predicting with model
        y_pr_ = gsCV.predict(X_)
        # Plotting histogram of predicted values
        short_lab = True if mod == 'class' else False
        if plot: plot_hist_pred_val(y_te, y_pr, y_pr_, short_lab=short_lab)
        # returning indexes to impute and calculated values
        return ind_to_impute, y_pr_


################### End of KNN Imputation ########################

# Plotting heatmap (2 options available, rectangle or triangle )

def plot_heatmap(corr, title, figsize=(8,4), vmin=-1, vmax=1, center=0,
                 palette = sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False):
    
    fig, ax = plt.subplots(figsize=figsize)
    if shape == 'rect':
        mask=None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    palette = palette
    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10},fmt=fmt,
                     square=False, linewidths=.5, linecolor = 'white',
                     cbar_kws={"shrink": .9, 'label': None}, robust = robust,
                     xticklabels= corr.columns, yticklabels = corr.index)
    ax.tick_params(labelsize=10,top=False, bottom=True,
                labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=12)


# Plotting explained variance ratio in scree plot

def scree_plot(col_names, exp_var_rat, ylim=(0,0.4)):
    plt.bar(x=col_names, height=exp_var_rat, color='grey')
    ax1 = plt.gca()
    ax1.set(ylim=ylim)
    ax2 = ax1.twinx()
    ax2.plot(exp_var_rat.cumsum(), 'ro-')
    ax2.set(ylim=(0,1.1))
    ax1.set_ylabel('explained var. rat.')
    ax2.set_ylabel('cumulative explained var. rat.')

    for i, p in enumerate(ax1.patches):
        ax1.text( p.get_width()/5 + p.get_x(), p.get_height()+ p.get_y()+0.01,
                '{:.0f}%'.format(exp_var_rat[i]*100),
                    fontsize=8, color='k')
        
    plt.gcf().set_size_inches(8,3)
    plt.title('Scree plot', fontweight='bold')


# Returing main regressor scores

def scores_reg(name, Xte, yte, ypr, adj_r2=False):
    # MAE = metrics.mean_absolute_error(yte, ypr)
    MSE = metrics.mean_squared_error(yte, ypr)
    RMSE = np.sqrt(MSE)
    R2 = metrics.r2_score(yte, ypr)
    n = yte.shape[0] # nb of observations
    p = Xte.shape[1] # nb of indep features
    if adj_r2:
        Adj_R2 = 1-(1-R2)*(n-1)/(n-p-1)
        return pd.Series([RMSE, R2, Adj_R2],
                        index = ['RMSE', 'R2', 'Adj_R2'],
                        name=name)
    else:
        return pd.Series([RMSE, R2],
                     index = ['RMSE', 'R2'],
                     name=name)
# Computing the Adjusted R2 score

from sklearn.model_selection import cross_validate

def calc_adj_R2(R2, n, p):
    # n = yte.shape[0] # n: nb of observations
    # p = Xte.shape[1] # p: nb of indep features
    return 1-(1-R2)*(n-1)/(n-p-1)


# Returning mean regressor scores with cross-validation

def cv_scores_reg(name, pipe, X, y, cv=5, adj_r2=False):
    res = pd.Series()
    cv_scoring = ['neg_root_mean_squared_error', 'r2']

    cv_scores = cross_validate(pipe, X, y, scoring=cv_scoring,                       
                               cv=cv, return_train_score=True, verbose=1)
    if adj_r2:
        res = pd.Series({'mean_CV_te_RMSE': -cv_scores['test_neg_root_mean_squared_error'].mean(),
	                     'mean_CV_te_R2': cv_scores['test_r2'].mean(),
	                     'mean_CV_te_adjR2': calc_adj_R2(cv_scores['test_r2'],
                                                      y.shape[0]/5,
                                                      X.shape[1]).mean()},
                        name = name)
    else:
	    res = pd.Series({'mean_CV_te_RMSE': -cv_scores['test_neg_root_mean_squared_error'],
	                     'mean_CV_te_R2': cv_scores['test_r2']}, name = name)
    return res


## computes the test score from fitted model and appends to current dataframe or create a new one

def get_append_scores(name_reg, pipe, Xte, yte, df_res=None, cv=6):
    if df_res is None:
        df_res = pd.DataFrame(dtype = 'object')
    df_res_mod = pd.DataFrame(dtype = 'object')
    ypr = pipe.predict(Xte)
    ser = scores_reg(name_reg, Xte, yte, ypr).astype('object')
    ser = ser.append(cv_scores_reg(name_reg, pipe, Xte, yte, cv=cv).astype('object'))
    df_res_mod = pd.concat([df_res,ser.to_frame()],1)
    return df_res_mod

''' Builds a customizable column_transformer which parameters can be optimized in a GridSearchCV
CATEGORICAL : three differents startegies for 3 different types of
categorical variables:
- low cardinality: customizable strategy (strat_low_card)
- high cardinality: customizable strategy (strat_high_card)
- boolean or equivalent (2 categories): ordinal
QUANTITATIVE (remainder): 
- StandardScaler

-> EXAMPLE (to use apart from gscv):
cust_enc = CustTransformer(thresh_card=12,
                       strat_binary = 'ord',
                       strat_low_card = 'ohe',
                       strat_high_card = 'loo',
                       strat_quant = 'stand')
cust_enc.fit(X_tr, y1_tr)
cust_enc.transform(X_tr).shape, X_tr.shape

'''

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.preprocessing import *

class CustTransformer(BaseEstimator) :

    def __init__(self, thresh_card=12,
                 strat_binary = 'ord', strat_low_card ='ohe',
                 strat_high_card ='hash', strat_quant = 'stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant

    def d_type_col(self, X):
        bin_cols = (X.nunique()[X.nunique()==2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = X_C_cols.nunique()[X_C_cols.nunique()<self.thresh_card].index
        C_h_card_cols = X_C_cols.nunique()[X_C_cols.nunique()>=self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns\
                                                        if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'remaind': Q_cols}
        return d_t

    def fit(self, X, y=None):
        # Dictionary to translate strategies
        d_enc = {'ohe': ce.OneHotEncoder(),
                 'hash': ce.HashingEncoder(),
                 'ord': ce.OrdinalEncoder(),
                 'loo': ce.LeaveOneOutEncoder(),
                 'bin': ce.BinaryEncoder(),
                 'stand': StandardScaler(),
                 'minmax': MinMaxScaler(),
                 'maxabs': MaxAbsScaler(),
                 'robust': RobustScaler(quantile_range=(25, 75)),
                 'norm': Normalizer(),
                 'quant_uni': QuantileTransformer(output_distribution='uniform'),
                 'quant_norm': QuantileTransformer(output_distribution='normal'),
                 'pow': PowerTransformer(method='yeo-johnson'), # 'boxcox'
                 }
        # Creates a columns transformer with chosen strategies
        self.column_trans = \
                ColumnTransformer([("binary", d_enc[self.strat_binary],
                                    self.d_type_col(X)['binary']),
                                   ("low_card", d_enc[self.strat_low_card],
                                    self.d_type_col(X)['low_card']),
                                   ("high_card", d_enc[self.strat_high_card],
                                    self.d_type_col(X)['high_card']),
                                   ("remaind", d_enc[self.strat_quant],
                                    self.d_type_col(X)['remaind'])])
        return self.column_trans.fit(X, y)
  
    def transform(self, X, y=None):
        return  self.column_trans.transform(X)
''' create a pipeline with datapreprocessing column transformer and regressor
    then searches for best hyperparameters with gscv
    then stores best parameters
    then computes the scores of the model on testing set
    then computes the cv scores of the model on testing set'''

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV

def model_optimizer(data_preproc, name_reg, reg, param_grid,
                    Xtr, ytr, Xte, yte,
                    cv_search=5, groups=None, cv_test=6,
                    gs_score='neg_root_mean_squared_error',
                    search_strat='grid', n_iter=10):

    pipe = Pipeline([('preproc', data_preproc),
                    (name_reg, reg)])
    
    # researching best hyperparameters and fitting on training set
    if search_strat=='grid':
        scv = GridSearchCV(pipe, param_grid = param_grid,
                           cv=cv_search, scoring=gs_score, verbose=1)
        print("grid")
    elif search_strat=='rand':
        scv = RandomizedSearchCV(pipe, param_distributions = param_grid,
                            cv=cv_search, n_iter= n_iter,
                            scoring=gs_score, verbose=1)
        print("randomized")
    else:
        print("ERROR: This strategy of hyperparameter tuning does not exist.")
    scv.fit(Xtr, ytr, groups=groups) # to stratify the folds using a GroupFolds

    # best hyperparams
    df_res = pd.DataFrame(dtype = 'object')
    df_res.at['name_params', name_reg] =\
                    str(list(param_grid.keys()))
    df_res.at['best_params', name_reg] =\
                    str([scv.best_params_[p] for p in param_grid])

    # score of the model with best params on testing set
    ypr = scv.predict(Xte)
    res = scores_reg(name_reg, Xte, yte, ypr).astype('object')
    df_res = df_res.append(res.to_frame())

    # mean cv score of the model with best params on testing set
    res = cv_scores_reg(name_reg, scv.best_estimator_,
                        Xte, yte, cv=cv_test).astype('object')
    df_res = df_res.append(res.to_frame())

    return scv, df_res

# def model_optimizer(data_preproc, name_reg, reg,  param_grid,
#                     Xtr, ytr, Xte, yte, cv_gs=5, cv_test=5):

#     param_grid_pipe = {str(reg)[:-2].lower()+'__'+k : v \
#                 for k,v in param_grid.items()}
#     pipe = make_pipeline(data_preproc, reg)

#     # researching best hyperparameters and fitting on training set
#     gscv = GridSearchCV(pipe,
#                         param_grid=param_grid_pipe,
#                         cv=cv_gs, verbose=1)
#     gscv.fit(Xtr,ytr)

#     # best hyperparams
#     df_res = pd.DataFrame(dtype = 'object')
#     df_res.at['name_params', name_reg] =\
#                     str(list(param_grid.keys()))
#     df_res.at['best_params', name_reg] =\
#                     str([gscv.best_params_[p] for p in param_grid_pipe])

#     # score of the model with best params on testing set
#     ypr = gscv.predict(Xte)
#     res = scores_reg(name_reg, Xte, yte, ypr).astype('object')
#     df_res = df_res.append(res.to_frame())

#     # mean cv score of the model with best params on testing set
#     res = cv_scores_reg(name_reg, gscv.best_estimator_,
#                         Xte, yte, cv=cv_test).astype('object')
#     df_res = df_res.append(res.to_frame())

#     return gscv, df_res

## When searching for 2 best hyperparameters with gscv : plotting a heatmap of mean_test_score(cv)

def plot_2D_hyperparam_opt(gscv, params=None):
    gscv_res = gscv.cv_results_
    df_gscv = pd.DataFrame(gscv_res)
    if params:
        params_gscv = ['param_'+p for p in params]  # example: params=['my_ElasticNet__alpha', 'my_ElasticNet__l1_ratio']
    else:
        params_gscv = df_gscv.columns[df_gscv.columns.str.contains('param_')].to_list()
        if len(params_gscv)!=2:
            print('ERROR : please provide exactly two parameters to display')
            return None
            # params_gscv = params_gscv[0:2]
        else:
            params_gscv = params_gscv
    max_scores = df_gscv.groupby(params_gscv).max()
    sns.heatmap(max_scores.unstack()['mean_test_score'], annot=True, fmt='.4g');

## When searching for 1 best hyperparameters with gscv : plotting a heatmap of mean_test_score(cv)

def plot_1D_hyperparam_opt(gscv, log_sc=False, param=None):
   
    gscv_res = gscv.cv_results_
    df_gscv = pd.DataFrame(gscv_res)
    if param:
        param_gscv = 'param_'+param  # example: param='KNN__n_neighbors'
    else:
        param_gscv = df_gscv.columns[df_gscv.columns.str.contains('param_')]
        if len(param_gscv)!=1:
            print('ERROR : there is more than one parameter, try smthg else')
            return None
        else:
            param_gscv = param_gscv[0]
    
    li_p = gscv.cv_results_[param_gscv].tolist()
    max_scores = df_gscv.groupby(param_gscv).max()
    plt.errorbar(li_p, max_scores.unstack()['mean_test_score'], color='r',
                yerr=gscv_res['std_test_score'])
    plt.gca().set_title(param_gscv)
    if log_sc: plt.gca().set_xscale('log')
    plt.gcf().set_facecolor('w')


''' Plotting the leraning curve of a model.
Allow iterative addition of other curves on the same figure if passed in arguments'''

from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, train_sizes, label, c='r',
                        scoring="r2", cv=5, fig=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax = fig.axes[0]
    train_sizes, train_scores,valid_scores = \
                    learning_curve(model, X, y, train_sizes=train_sizes,
                                   scoring=scoring, cv=cv)
    scores = -valid_scores.mean(1) if scoring!='r2' else valid_scores.mean(1)
    ax.plot(train_sizes, scores, 'o-', color=c, label=label)
    ax.set_xlabel("Train size"), ax.set_ylabel("R2")
    ax.set_title('Learning curves', fontweight='bold')
    ax.legend(loc="best")
    fig.set_facecolor('w')
    return fig


'''calculates VIF and exclude colinear columns'''

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def select_from_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + str(X.iloc[:, variables].columns[maxloc]) +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def select_from_vif_debugged_(X, thresh=100):
	cols = X.columns
	variables = np.arange(X.shape[1])
	dropped=True
	while dropped:
	    dropped=False
	    c = X[cols[variables]].values
	    vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

	    maxloc = vif.index(max(vif))
	    if max(vif) > thresh:
	        print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
	        variables = np.delete(variables, maxloc)
	        dropped=True

	print('Remaining variables:')
	print(X.columns[variables])
	return X[cols[variables]]