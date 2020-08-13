import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import seaborn as sns

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn import metrics

def move_cat_containing(my_index, strings, order='last'):
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

import scipy.stats as st

def plot_histograms(df, cols, file_name=None, bins=30, figsize=(12,7), skip_outliers=True, thresh=3, layout=(3,3), save_enabled=False):

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        if skip_outliers:
            ser = df[c][np.abs(st.zscore(df[c]))<thresh]
        else:
            ser = df[c]
        ax.hist(ser,  bins=bins, color='grey')
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))
        
    plt.tight_layout(w_pad=0.5, h_pad=0.65)

    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400);
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


''' computes the scores from true values, predicted values
and returns the result in a pd.Series'''

def scores_reg(name_reg, y, ypr, X=None, scoring_test=None, exclude=['Adj_R2']):
  
    MAE = metrics.mean_absolute_error(y, ypr)
    MSE = metrics.mean_squared_error(y, ypr)
    RMSE = np.sqrt(MSE)
    R2 = metrics.r2_score(y, ypr)
    n = y.shape[0] # nb of observations
    p = X.shape[1] # nb of indep features
    Adj_R2 = 1-(1-R2)*(n-1)/(n-p-1)
    MAPE = 100*np.mean(np.abs((y-ypr)/(y+1e-10)))
    MSPE = 100*np.mean(np.square((y-ypr)/y+1e-10))
    expl_var = metrics.explained_variance_score(y, ypr)

    scoring_test_res = {'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE,
                    'R2': R2, 'expl_var': expl_var, 'Adj_R2': Adj_R2,
                    'MAPE': MAPE, 'MSPE': MSPE}

    li_n_metrics = [n for n in scoring_test_res.keys() if n not in exclude]
    li_metrics = [scoring_test_res[n] for n in li_n_metrics]
    ser = pd.Series(li_metrics, index = li_n_metrics, name=name_reg)

    return ser

''' Computes predictions using a given model,
and then computes and returns the given scores
# 2 | score of the model with best params on a given test set'''

def compute_test_scores(name_reg, model, X, y, scoring_test=None, exclude=['Adj_R2']):
    
    df_res = pd.DataFrame(dtype = 'object')
    if scoring_test is None:  # default scores
        ypr = model.predict(X)
        ser = scores_reg(name_reg, y, ypr, X, scoring_test, exclude).astype('object')
    else: # attention un score fait avec make_scorer prend en entrée (estimator, X, y), et pas (yte, ypr)
        li_metrics = [scoring_test[n](model, X, y) for n in scoring_test.keys()]
        ser = pd.Series(li_metrics, index = scoring_test.keys(), name=name_reg)
    df_res = df_res.append(ser.to_frame())
    return df_res

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

-> EXAMPLE (to fetch names of the modified dataframe):
small_df = df[['Outlier', 'Neighborhood', 'CertifiedPreviousYear',
               'NumberofFloors','ExtsurfVolRatio']]
# small_df.head(2)
cust_trans = CustTransformer()
cust_trans.fit(small_df)
df_enc = cust_trans.transform(small_df)
cust_trans.get_feature_names(small_df)

'''

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.preprocessing import *

class CustTransformer(BaseEstimator) :

    def __init__(self, thresh_card=12,
                 strat_binary = 'ord', strat_low_card ='ohe',
                 strat_high_card ='bin', strat_quant = 'stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant

    def d_type_col(self, X):
        bin_cols = (X.nunique()[X.nunique()<=2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = X_C_cols.nunique()[X_C_cols.nunique().between(3, self.thresh_card)].index
        C_h_card_cols = X_C_cols.nunique()[X_C_cols.nunique()>self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns\
                                                        if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        return d_t

    def get_feature_names(self):
        return self.ct_cat.get_feature_names()+self.num_cols

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
                 'boxcox': PowerTransformer(method='boxcox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'none': FunctionTransformer(func=lambda x:x,
                                             inverse_func=lambda x:x),
                 }

        self.ct_cat = ColumnTransformer([
                                        ('binary', d_enc[self.strat_binary], self.d_type_col(X)['binary']),
                                        ('low_card', d_enc[self.strat_low_card], self.d_type_col(X)['low_card']),
                                        ('high_card', d_enc[self.strat_high_card], self.d_type_col(X)['high_card']),
                                        ],
                                        #remainder='passthrough'
                                        )

        self.num_cols = self.d_type_col(X)['numeric']
        self.num_trans = Pipeline([("numeric", d_enc[self.strat_quant])])

        self.cat_cols = self.d_type_col(X)['binary'].union(self.d_type_col(X)['low_card']).union(self.d_type_col(X)['high_card'])
        self.cat_trans = Pipeline([("categ", self.ct_cat)])

        self.column_trans =  ColumnTransformer([
                                        ('cat', self.cat_trans, self.cat_cols),
                                        ('num', self.num_trans, self.num_cols),
                                        ],
                                        # remainder='passthrough'
                                        )
        self.ct_cat.fit(X, y)
        return self.column_trans.fit(X, y)
  
    def transform(self, X, y=None): # to use in a pipeline
        return  self.column_trans.transform(X)

    def transform_df(self, X, y=None): # to get a dataframe
        return pd.DataFrame(self.column_trans.transform(X),
                            columns=self.get_feature_names())

    def fit_transform(self, X, y=None):
        if y is None:
            self.fit(X)
            return self.column_trans.transform(X)
        else:
            self.fit(X, y)
            return self.column_trans.transform(X)

    def fit_transform_df(self, X, y=None):
        if y is None:  
            self.fit(X)
            return pd.DataFrame(self.column_trans.transform(X),
                            columns=self.get_feature_names())
        else: 
            self.fit(X, y)
            return pd.DataFrame(self.column_trans.transform(X,y),
                            columns=self.get_feature_names())

''' Class to filter outliers from X and y from the zscore of the X columns
eliminates a line if the value of one or more features is outlier. 
(CANNOT BE USED IN A PIPELINE !!!)'''

import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin

class ZscoreSampleFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thresh = None):
        self.thresh = thresh

    def fit(self, X, y=None):
        self.X_zscore = X.apply(st.zscore, axis=0)
        # self.featurefilter = [True for c in X.columns] # pas de filtrage de colonnes
        self.samplefilter = (np.abs(self.X_zscore)<self.thresh).all(1) # filtrage de lignes
        return self

    def transform(self, X, y=None, copy=None):
        # X_mod = X.loc[:,self.featurefilter]
        X_mod = X.loc[self.samplefilter]
        if y is not None:
            y_mod = y.loc[self.samplefilter]
            return X_mod, y_mod
        else:
            return X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X,y)

"""  
Class to filter outliers from X and y from a LOF analysis on X
(CANNOT BE USED IN A PIPELINE !!!)
A threshold is set for selection criteria, 
neg_conf_val (float): threshold for excluding samples with a lower
 negative outlier factor.
 NB: may not be that useful, because we can use LocalOutlierFactor.predict method...
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor

class LOFSampleFilter(BaseEstimator, TransformerMixin):
    def __init__(self, neg_conf_val=None, n_neighbors=None, **kwargs):
        self.neg_conf_val = neg_conf_val if neg_conf_val is not None else -10
        self.n_neighbors = n_neighbors if n_neighbors is not None else 20
        self.kwargs = kwargs

    def fit(self, X, y=None, *args, **kwargs):
        # X = np.asarray(X)
        # y = np.asarray(y)
        lcf = LocalOutlierFactor(n_neighbors=self.n_neighbors, **self.kwargs)
        lcf.fit(X)
        # self.featurefilter = [True for c in X.columns] # pas de filtrage de colonnes
        self.samplefilter = lcf.negative_outlier_factor_ > self.neg_conf_val # filtrage de lignes
        return self

    def transform(self, X, y=None, copy=None):
        # X_mod = X.loc[:,self.featurefilter]
        X_mod = X.loc[self.samplefilter]
        if y is not None:
            y_mod = y.loc[self.samplefilter]
            return X_mod, y_mod
        else:
            return X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X,y)

"""  
Class to filter outliers from X and y from a LOF analysis on X
(CANNOT BE USED IN A PIPELINE !!!)
A threshold is set for selection criteria, 
score_samples (float): threshold for excluding samples with a lower
score_samples.
NB: may not be that useful, because we can use IsolationForest.predict method...
"""

from sklearn.ensemble import IsolationForest

class IsolForestSampleFilter(BaseEstimator, TransformerMixin):
    def __init__(self, score_samples=None, n_estimators=None, **kwargs):
        self.score_samples = score_samples if score_samples is not None else 2
        self.n_estimators = n_estimators if n_estimators is not None else 100
        self.kwargs = kwargs

    def fit(self, X, y=None, *args, **kwargs):
        # X = np.asarray(X)
        # y = np.asarray(y)
        isolf = IsolationForest(n_estimators=self.n_estimators, **self.kwargs)
        isolf.fit(X)
        # self.featurefilter = [True for c in X.columns] # pas de filtrage de colonnes
        self.samplefilter = isolf.score_samples(X) > self.score_samples # filtrage de lignes
        return self

    def transform(self, X, y=None, copy=None):
        # X_mod = X.loc[:,self.featurefilter]
        X_mod = X.loc[self.samplefilter]
        if y is not None:
            y_mod = y.loc[self.samplefilter]
            return X_mod, y_mod
        else:
            return X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X,y)

##### Function to get the type of columns before encoding

def d_type_col(X, thresh_card=12):
    bin_cols = (X.nunique()[X.nunique()<=2].index)
    X_C_cols = X.select_dtypes(include=['object', 'category'])
    C_l_card_cols = X_C_cols.nunique()[X_C_cols.nunique().between(3, thresh_card)].index
    C_h_card_cols = X_C_cols.nunique()[X_C_cols.nunique()>thresh_card].index
    Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns\
                                                    if c not in bin_cols]
    d_t = {'binary': bin_cols,
            'low_card': C_l_card_cols,
            'high_card': C_h_card_cols,
            'numeric': Q_cols}
    return d_t

''' create a pipeline with datapreprocessing column transformer and regressor
    then searches for best hyperparameters with gscv
    then stores best parameters
    then computes the scores of the model on testing set
    then computes the cv scores of the model on testing set'''

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV

#from sklearn_pandas import DataFrameMapper

def model_optimizer(name_reg, reg, param_grid,
                    pipe, X=None, y=None, 
                    scv_scores='neg_root_mean_squared_error',
                    refit='neg_root_mean_squared_error', cv_search=5,
                    search_strat='grid', n_iter=10, groups=None, verbose=1):
    
    if search_strat=='grid':
        scv = GridSearchCV(pipe, param_grid = param_grid,
                           cv=cv_search, 
                           scoring=scv_scores, 
                           refit=refit,
                           return_train_score=True,
                           verbose=1)
        print("Grid")
    elif search_strat=='rand':
        scv = RandomizedSearchCV(pipe, param_distributions = param_grid,
                                 cv=cv_search,
                                 scoring=scv_scores, 
                                 refit=refit,
                                 return_train_score=True,
                                 n_iter=n_iter, verbose=1)
        print("Randomized")
    else:
        print("ERROR: This strategy of hyperparameter tuning doesn't exist.")

    scv.fit(X, y, groups=groups) # to use a GroupKFolds

    return scv


''' Function that encapsulates model_optimizer,
test of wether the model already exists or not in pickle,
saving the computed model in the pickle,
aggregate the results in a dataframe'''

import dill

def run_optimization(name_reg, reg, param_grid, file_name, dict_models, pipe, dict_scv_params,
	                 cv_search, df_res, search_strat, n_iter=50):

    # If model with the same name already in dict_models, just takes existing model
    if dict_models.get(name_reg, np.nan) is not np.nan:
        print('-----Model already exists - taking existing model')
    # Else computes new model and add to the dictionnary, and then to the pickle
    else:
        print('-----Model not existing - computing...')
        dict_models[name_reg] = \
            model_optimizer(name_reg, reg, param_grid, pipe, **dict_scv_params, cv_search=cv_search,
                            search_strat=search_strat, n_iter=n_iter)
    with open(file_name, "wb") as f:
        dill.dump(dict_models, f)
    print("-----...model dumped")
    
    new_df_res = scv_perf_fetcher(name_reg, dict_models[name_reg])
    return pd.concat([df_res, new_df_res], axis=1)


# Recreates the encoded and scaled DataFrame

def get_best_model_transf_df(scv, X, y):
    best_c_trans = scv.best_estimator_.named_steps['preproc']
    best_c_trans.fit(X, y)
    name_cols = best_c_trans.get_feature_names(X)
    X_enc = pd.DataFrame(best_c_trans.transform(X),
                            columns=name_cols,
                            index =X.index)
    return X_enc

''' 1 | fetches best hyperparams and scores obtained
during searchCV (training, testing)'''

def scv_perf_fetcher(name_reg, scv):
    
    df_res = pd.DataFrame(dtype = 'object')
    df_best_res = pd.DataFrame(scv.cv_results_).loc[scv.best_index_].astype('object')
    df_res.at['best_params', name_reg] = str(df_best_res['params'])
    li_index = df_best_res.index[df_best_res.index.str.startswith(('mean_', 'std_'))]
    li_index = move_cat_containing(li_index, ['train', 'test', 'score', 'fit'])
    for i in li_index:
        df_res.at[i, name_reg] = df_best_res[i]

    return df_res
    
## When searching for 2 best hyperparameters with gscv : plotting a heatmap of mean_test_score(cv)
## the score displayed for each cell is the one for the best other parameters.

def plot_2D_hyperparam_opt(scv, params=None, score = 'neg_root_mean_squared_error',
                           title=None, ax=None):

    scv_res = scv.cv_results_
    df_scv = pd.DataFrame(scv_res)
    if params: # example: params=['enet__alpha', 'enet__l1_ratio']
        params_scv = ['param_'+p for p in params]
    else:
        params_scv = df_scv.columns[df_scv.columns.str.contains('param_')].to_list()
        if len(params_scv)!=2:
            print('WARNING : parameters to display were guessed,\
                provide the params parameter with 2 parameters')
            params_scv = params_scv[0:2]
        else:
            params_scv = params_scv
    # Not suitable for 3D viz : takes the max among all other parameters !!!
    max_scores = df_scv.groupby(params_scv).agg(lambda x: max(x))
    sns.heatmap(max_scores.unstack()['mean_test_'+score],
                annot=True, fmt='.4g', ax=ax);
    # A CHANGER PEUT-ETRE EVENTUELLEMENT POUR AFFICHER DES NOMBRES COMPACTS, QUAND LES PARAMS SONT DES NOMBRES
    # print(plt.gca().get_yticklabels()[1].get_text())
    # plt.gca().set_xticklabels = ['{:.4}'.format(x) if type(x)==np.number else x for x in plt.gca().get_xticklabels() ]
    # plt.gca().set_yticklabels = ['{:.4}'.format(y) if type(y)==np.number else y for y in plt.gca().get_yticklabels() ]
    if title is None:  title = score
    plt.gcf().suptitle(title)

'''
Generate 3 plots: the test and training learning curve, the training
samples vs fit times curve, the fit times vs score curve.
'''

from sklearn.model_selection import ShuffleSplit
from matplotlib.lines import Line2D

def plot_learning_curve(name_reg, estimator, X, y, ylim=None, cv=None,
                        scoring='neg_root_mean_squared_error', score_name = "Score",
                        file_name=None, dict_learn_curves=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        c='r', axes=None, title=None):
    if axes is None : fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # plt.subplots(0, 3, figsize=(12, 3))

    if dict_learn_curves is None: dict_learn_curves = {}

    # If model with the same name already in dict_models, just takes existing model
    if dict_learn_curves.get(name_reg, np.nan) is not np.nan:
        print('-----Learning curve already exists - taking existing learning curve')
        train_sizes, train_scores, test_scores, fit_times = \
                         list(zip(*list(dict_learn_curves[name_reg].items())))[1]
    
    # Else computes new model and add to the dictionnary, and then to the pickle
    else:
        print('----- Learning curve not existing - computing...')

        train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                     train_sizes=train_sizes, scoring = scoring,
                       return_times=True) 
        
        d_ = {'train_sizes': train_sizes,
          'train_scores': train_scores,
          'test_scores': test_scores,
          'fit_times': fit_times}
        dict_learn_curves[name_reg] = d_
        if file_name is not None:
            with open(file_name, "wb") as f:
                dill.dump(dict_learn_curves, f)
            print("-----...learning curve dumped")
        else:
            print("-----...no file name to dump the learning curves dictionary")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.15,
                         color=c)
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.3,
                         color=c)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', mfc=None, ms=3,
                 color=c, ls = 'dashed',  label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', ms=3,
                 color=c, ls = 'solid',
                 label="Cross-validation score")
    axes[0].set_title("Learning curves")
    if ylim is not None: axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(score_name)
    
    cust_leg = [Line2D([0], [0], color='k', ls = 'dashed', lw=2),
                Line2D([0], [0], color='k', ls = 'solid', lw=2)]
    axes[0].legend(cust_leg, ['Train (CV)', 'Test (CV)'],loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean,
                 'o-', color=c, ms=3)
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, color=c, alpha=0.2)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean,
                 'o-', ms=3, color=c, label=name_reg)
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, color=c, alpha=0.2)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel(score_name)
    axes[2].set_title("Performance of the model")
    axes[2].legend(loc=2, prop={'size': 10})# bbox_to_anchor = (0.2,1.1), ncol=4

    plt.gcf().set_facecolor('w')
    if title is not None:
        plt.gcf().suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout(rect=(0,0,1,0.92))
    else:
        plt.tight_layout()
    return plt

''' Plotting the leraning curve of a model.
Allow iterative addition of other curves on the same figure if passed in arguments'''

from sklearn.model_selection import learning_curve

# def plot_learning_curve_old_version(model, X, y, train_sizes, label, c='r',
#                         scoring="r2", cv=5, fig=None):
#     if fig is None:
#         fig = plt.figure()
#         ax = fig.add_subplot()
#     ax = fig.axes[0]
#     train_sizes, train_scores, valid_scores = \
#                     learning_curve(model, X, y, train_sizes=train_sizes,
#                                    scoring=scoring, cv=cv)
#     scores = -valid_scores.mean(1) if scoring!='r2' else valid_scores.mean(1)
#     ax.plot(train_sizes, scores, 'o-', color=c, label=label)
#     ax.set_xlabel("Train size"), ax.set_ylabel(scoring)
#     ax.set_title('Learning curves', fontweight='bold')
#     ax.legend(loc="best")
#     fig.set_facecolor('w')
#     return fig


''' Plots the training and test scores obtained during the SearchCV (either Randomized or Grid)
the other parameters are parameters of the best estimator (found by gridsearch)'''


def plot_scv_multi_scores(name_reg, scv, param, title = None, x_log=False, loc='best', figsize = (12, 4)):

    if name_reg is None :
        name_reg = scv.estimator.steps[2][0]

    best_params, df_sel, df_gscv_filt = filters_cv_results(scv,param)
    results = df_gscv_filt

    scoring = scv.scoring
    fig, axs = plt.subplots(1,len(scoring))
    fig.set_size_inches(figsize)
    
    li_colors = ['b', 'r', 'g', 'purple', 'orange', 'pink']
    if len(axs)==1 : axs = [axs]

    # Get the regular np array from the MaskedArray
       
    X_axis = np.array(results['param_'+param], dtype='float')
        
    for scorer, color, ax in zip(sorted(scoring), li_colors[:len(scoring)], axs):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)].values
            sample_score_std = results['std_%s_%s' % (sample, scorer)].values
            alpha = 0.2 if sample == 'test' else 0.1
            df_ = pd.DataFrame({'param': X_axis,
                                'mean': sample_score_mean,
                                'std': sample_score_std}).sort_values(by='param')
            ax.fill_between(df_['param'],
                            df_['mean'] - df_['std'],
                            df_['mean'] + df_['std'],
                            alpha=alpha, color=color)
            ax.plot(df_['param'], df_['mean'], style, marker='o', markersize=3,
                color=color, alpha=1 if sample == 'test' else 0.7, label=f"{sample}")
            if x_log: ax.set_xscale('log')
            ax.set_title(scorer)
            
        y_min, y_max = ax.get_ylim()
        
        # Plot a dotted vertical line at the best score for that scorer marked by x
        best_index = results['rank_test_%s' % scorer].argmin()
        best_score = results['mean_test_%s' % scorer].iloc[best_index]
        ax.plot([X_axis[best_index], ] * 2, [y_min - abs(y_min)*0.1, best_score],
            linestyle='dotted', color=color, marker='x', markeredgewidth=3, ms=8)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(param)
        ax.set_ylabel("Score")
        ax.legend(loc=loc)

        # Annotate the best score for that scorer
        len_str = len("{:.2f}".format(best_score))
        if X_axis[best_index] < np.mean(X_axis):
            x_pos = X_axis[best_index]*(1+0.015*len_str)
        else:
            x_pos = X_axis[best_index]*(1-0.015*len_str)
        y_pos = best_score*1+(y_max-y_min)*0.05
        ax.annotate("{:0.2f}".format(best_score), 
                    (x_pos, y_pos),
                    color = color)  
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0,0,1,0.92))
    else:
        plt.tight_layout()
    plt.show()

''' Takes a gridsearch or randomizedsearch and one parameter
and isolate the influence of this parameter on all the scores
available in the scv -> dictionary of the best other fixed parameters 
and a dataframe of the scores depending on the chosen parameter and a 
filtered cv_results_ dataframe that can be used in the 'plot_scv_multi_scores' function '''

def filters_cv_results(gscv, param):

    gscv_res = gscv.cv_results_
    df_gscv = pd.DataFrame(gscv_res)
    param_gscv = 'param_'+param  # example: param='KNN__n_neighbors'

    # selects in the data frame the best params except for 'param' (les afficher en titre ??)
    all_params = df_gscv.columns[df_gscv.columns.str.startswith('param_')]
    all_params_sh = [p[6:] for p in all_params]
    best_params = gscv.best_params_.copy()
    del best_params[param]

    # filters in the result dataframe only optimized results except for 'param'
    mask = np.full((df_gscv.shape[0],), True)
    for k,v in best_params.items():
        mask = mask & (df_gscv['param_'+k]==v)

    df_gscv_filt = df_gscv.loc[mask]

    li_p = df_gscv_filt[param_gscv].tolist()

    scores = df_gscv_filt.columns[df_gscv_filt.columns.str.startswith('mean_')]
    df_sel_scores = df_gscv_filt[['param_'+param]+list(scores)].set_index('param_'+param)

    return best_params, df_sel_scores, df_gscv_filt

'''Plotting one given score for all or a selection of the hyperparameters tested with a gsearch
Can choose the aggregation function for the score on all other parameters
option for using pooled standard deviation in stead of regular std'''

def plot_hyperparam_tuning(gs, grid_params, params=None, score='score',
                           pooled_std=False, agg_func=np.mean):

    if params is not None:
        grid_params = {k:v for (k,v) in grid_params.items() if k in params}

    def pooled_var(stds):
        n = 5 # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))
    # recalculates the standard deviation using pooled variance
    std_func = pooled_var if pooled_std else np.std

    df = pd.DataFrame(gs.cv_results_)
    results = ['mean_test_'+score,
                'mean_train_'+score,
                'std_test_'+score, 
                'std_train_'+score]

    fig, axes = plt.subplots(1, len(grid_params), 
                            figsize = (3.2*len(grid_params), 3),
                            sharey='row')
    axes[0].set_ylabel(score, fontsize=12)

    for idx, (param_name, param_range) in enumerate(grid_params.items()):
        grouped_df = df.groupby('param_'+param_name)[results]\
            .agg({'mean_train_'+score: agg_func,
                'mean_test_'+score: agg_func,
                'std_train_'+score: std_func,
                'std_test_'+score: std_func})
        previous_group = df.groupby(f'param_{param_name}')[results]
        lw = 2
        axes[idx].plot(param_range, grouped_df['mean_train_'+score], label="Train (CV)",
                    color="darkorange",marker='o',ms=3, lw=lw)
        axes[idx].fill_between(param_range,
                            grouped_df['mean_train_'+score] - grouped_df['std_train_'+score],
                            grouped_df['mean_train_'+score] + grouped_df['std_train_'+score], 
                            alpha=0.2, color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_'+score],
                    label="Test (CV)", marker='o',ms=3, color="navy", lw=lw)
        axes[idx].fill_between(param_range,
                            grouped_df['mean_test_'+score] - grouped_df['std_test_'+score],
                            grouped_df['mean_test_'+score] + grouped_df['std_test_'+score],
                            alpha=0.2, color="navy", lw=lw)
        axes[idx].set_xlabel(param_name, fontsize=12)
        ymin, ymax = axes[idx].get_ylim()
        # axes[idx].set_ylim(ymin, 0*ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Hyperparameters tuning', x=0.4, y=0.95, fontsize=15, fontweight='bold')
    fig.legend(handles, labels, loc=1, ncol=1, fontsize=12)

    fig.subplots_adjust(bottom=0.25, top=0.85, right=0.97)  
    plt.show()

'''permutation importance using sklearn '''
from sklearn.inspection import permutation_importance

def plot_perm_importance(model, name_reg, X, y, scoring='r2',
                         dict_perm_imp=None, file_name=None):

    # If model with the same name already in dict_models, just takes existing model
    if dict_perm_imp.get(name_reg, np.nan) is not np.nan:
        print('-----Permutation importance - taking existing model')
        ser = dict_perm_imp[name_reg]
    # Else computes new model and add to the dictionnary, and then to the pickle
    else:
        print('-----Permutation importance not existing - computing...')
        results = permutation_importance(model, X, y, scoring=scoring)
        ser = pd.Series(results.importances_mean, index = X.columns)
        
    dict_perm_imp[name_reg] = ser

    with open(file_name, "wb") as f:
        dill.dump(dict_perm_imp, f)
    print("-----...model dumped")

    fig, ax = plt.subplots()
    ser.sort_values(ascending=False).plot.bar(color='grey');
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )
    fig.set_size_inches(12,3)
    plt.show()

    return dict_perm_imp

'''Plotting the feature importance of a model'''

def plot_model_feat_imp(name_reg, model):
    
    # Getting the names of the transformed columns
    step_ct = model.named_steps['preproc'].named_steps['cust_trans']
    col_names = step_ct.get_feature_names()
    # Getting the list of the coefficients (wether usinf 'coef_' or 'feature_importances')
    step_reg = model.named_steps[name_reg]
    if hasattr(step_reg, "coef_"):
        col_coefs = step_reg.coef_
    elif hasattr(step_reg, "feature_importances_"):
        col_coefs = step_reg.feature_importances_
    else:
        print("ERROR: This regressor has no 'coef_' or 'feature_importances_' attribute")

    ser = pd.Series(col_coefs, index = col_names)

    fig, ax = plt.subplots()
    ser.sort_values(ascending=False).plot.bar(color='red');
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )
    fig.set_size_inches(15,3)
    plt.show()


''' Computing and plotting the regularisation path of a ridge, lasso or elasticnet model.
The parameters for encoding of the features must be passed.
 Browsing a given interval of alphas.
 (NOT YET POSSIBILITY TO ENTER OTHER BEST PARAMS OF THE REGRESSOR ITSELF)
'''
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from matplotlib.rcsetup import cycler

def plot_compute_reg_path(d_preproc, X, y, type_reg, alphas=np.logspace(-7,7,20)):

    preproc_pipe = Pipeline([('cust_trans', CustTransformer(**d_preproc))])
   
    ### NE PAS OUBLIER DE METTRE TOUTES LES AUTRES ETAPES SI NECESSAIRE !!!!!

    if type_reg == 'ridge':
        reg = Ridge()
    elif type_reg =='lasso':
        reg = Lasso()
    elif type_reg == 'enet':
        reg = ElasticNet()

    coefs = []
    for a in alphas:
        pipe_ = Pipeline([('preproc', preproc_pipe),
                        ('reg', reg.set_params(alpha=a, fit_intercept=False))])
        pipe_.fit(X, y)
        fitted_reg = pipe_.named_steps['reg']
        coefs.append(fitted_reg.coef_)

    print(len(alphas), len(coefs))

    fig, ax = plt.subplots()
    ## Plotting the regularization path (HOW CAN I MAKE THE CYCLER WORK ?)
    # default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
    #               cycler(linestyle=['-', '--', ':', '-.']))
    # with plt.rc_context(rc = {'axes.prop_cycle': default_cycler}):
        # ax.plot(alphas, coefs)
    ax.plot(alphas, coefs)
    # # Plotting a line for the best alpha PEUT ETRE REMETTRE UN GS ICI ?
    # best_alpha = ???????
    # ax.vlines(best_alpha, np.min(np.ravel(coefs)), np.max(np.ravel(coefs)))
    # Getting the names of the transformed columns
    step_ct = pipe_.named_steps['preproc'].named_steps['cust_trans']
    col_names = step_ct.get_feature_names()

    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    ax.legend(labels=col_names, ncol=2, bbox_to_anchor=[1,1])
    plt.xlabel('alpha'), plt.ylabel('weights')
    plt.title(type_reg.title()+' coefficients as a function of the regularization')
    plt.axis('tight')
    fig.set_size_inches(12,6)
    plt.show()

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

	''' SCORERS'''

from sklearn import metrics

# Mean Absolute Error
def calc_mae(y, ypr):
    return metrics.mean_absolute_error(y, ypr)
def calc_mae_log(y_log, y_log_pr):
    y = np.exp(y_log)-1
    ypr = np.exp(y_log_pr)-1
    return metrics.mean_absolute_error(y, ypr)

# Root Mean Squared Error
def calc_rmse(y, ypr):
    return np.sqrt(metrics.mean_squared_error(y, ypr))
def calc_rmse_log(y_log, y_log_pr):
    y = np.exp(y_log)-1
    ypr = np.exp(y_log_pr)-1
    return np.sqrt(metrics.mean_squared_error(y, ypr))

# Mean Percent Squared Error
def calc_mpse(y, ypr):
    return np.mean(np.square((y - ypr)/y))*100
def calc_mpse_log(y_log, y_log_pr):
    y = np.exp(y_log)-1
    ypr = np.exp(y_log_pr)-1
    return np.mean(np.square((y - ypr)/y))*100 

# R2 score
def calc_r2(y, ypr):
    return metrics.r2_score(y, ypr)
def calc_r2_log(y_log, y_log_pr):
    y = np.exp(y_log)-1
    ypr = np.exp(y_log_pr)-1
    return metrics.r2_score(y, ypr)

# Rate of predictions in 90-110%
def calc_pred_rate_10(y, ypr):
    return np.sum(np.abs((y - ypr)/y)<0.1)/y.size
def calc_pred_rate_10_log(y_log, y_log_pr):
    y = np.exp(y_log)-1
    ypr = np.exp(y_log_pr)-1
    return np.sum(np.abs((y - ypr)/y)<0.1)/y.size

mae = metrics.make_scorer(calc_mae, greater_is_better=False)
rmse = metrics.make_scorer(calc_rmse, greater_is_better=False) 
mpse = metrics.make_scorer(calc_mpse, greater_is_better=False)  
r2 = metrics.make_scorer(calc_r2, greater_is_better=True)
pred_rate_10 = metrics.make_scorer(calc_pred_rate_10, greater_is_better=True)

mae_log = metrics.make_scorer(calc_mae_log, greater_is_better=False)
rmse_log = metrics.make_scorer(calc_rmse_log, greater_is_better=False)
mpse_log = metrics.make_scorer(calc_mpse_log, greater_is_better=False)  
r2_log = metrics.make_scorer(calc_r2_log, greater_is_better=True) 
pred_rate_10_log = metrics.make_scorer(calc_pred_rate_10_log, greater_is_better=True)

'''Building and returning the dict_scv_params depending on the target and
the use of log transformation, as well as the names of the files to save scv and learning curves data.
'''
def set_dict_scv_params(X, y, target, log_on, refit):

    # Choice of the target
    if target == 'SEU':
        models_file_name = os.getcwd()+'/P4_models_SEU.pkl'
        l_curves_file_name = os.getcwd()+'/P4_lcurves_SEU.pkl'
        perm_imp_file_name = os.getcwd()+'/P4_pimp_SEU.pkl'
    elif target == 'GHG':
        models_file_name = os.getcwd()+'/P4_models_GHG.pkl'
        l_curves_file_name = os.getcwd()+'/P4_lcurves_GHG.pkl'
        perm_imp_file_name = os.getcwd()+'/P4_pimp_GHG.pkl'

    # Choice to fit y or log(1+y)
    if log_on: # scores defined in P4_functions.py
        y_mod = np.log1p(y.values)
        scorers = {'r2_log': r2_log,
                   'mae_log': mae_log,
                   'rmse_log': rmse_log,
                   'mpse_log': mpse_log,
                   'pred_rate_10_log': pred_rate_10_log}
        score_refit = refit+'_log'
    else:
        y_mod = y.values
        scorers = {'r2': r2,
                   'mae': mae,
                   'rmse': rmse,
                   'mpse': mpse,
                   'pred_rate_10': pred_rate_10}
        score_refit = refit

    dict_scv_params = dict(X = X,
                       y = y_mod,
                       scv_scores = scorers,
                       refit = score_refit)
    return dict_scv_params, models_file_name, l_curves_file_name, perm_imp_file_name


def load_pickle(f_name):
# If file of models exists, open and load in dict_model
    if os.path.exists(f_name):
        with open(f_name, "rb") as f:
            d_file = dill.load(f)
        print('--Pickle containing models already existing as ',
              f_name, ':\n', d_file.keys())
        print("Content loaded from '", f_name, "'.")
        return d_file
    # Else create an empty dictionary
    else:
        print('--No pickle yet as ',f_name)
        return {}