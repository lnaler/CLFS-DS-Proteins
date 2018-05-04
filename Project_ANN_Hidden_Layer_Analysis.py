import sys
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ImageMagickFileWriter
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

''' Make sure to run with command-line argument save when not debugging '''

plt.rcParams['animation.convert_path']='C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe'
mpl.verbose.set_level("helpful")

fig = plt.figure(figsize = (8,8))
ax = fig.gca(projection='3d')
max_vals_x = np.array([])
max_vals_y = np.array([])
mean_x_fr = 0
index_seen = []

###############################################################################
def update(i):
    '''Updates the plots for the gif to the next set of features being tested '''
    global max_vals_x
    global max_vals_y
    global mean_x_fr
    global index_seen
    
    feats = i*5+7
    label = 'Number of Features: {0}'.format(feats)
    
    sub_feats = features[:,np.random.choice(77,feats,replace=False)]
    temp_x = 0
    score_set = {}
    cv_all = []
    cv = []

    feat_step = int(feats/5)
    if(feat_step > 5):
        feat_step = 5
    
    for x in range (2,feats+1,feat_step):
        cv = [] 
        for y in range (2,feats+1,feat_step):
            classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (x,y))
            scores = cross_val_score(classifier, sub_feats, labels, cv = 10)
            cv.append(scores.mean())
            score_set[scores.mean()] = (x,y)
        cv_all.append(cv)
        
    x = range(2,feats+1,feat_step)
    y = range(2,feats+1,feat_step)
    x, y = np.meshgrid(x, y)
    cv_all = np.transpose(np.array(cv_all))

    if(i not in index_seen):
        ax.clear()
        print("i: ", i)
        regr= linear_model.LinearRegression()
        idx_end = len(score_set.keys())
        idx_beg = idx_end-5
        top_5 = sorted(score_set.keys())[idx_beg:idx_end]

        for j in range(5):
            temp_x += score_set[top_5[j]][0]
            max_vals_x = np.append(max_vals_x, score_set[top_5[j]][0])
            max_vals_y = np.append(max_vals_y, score_set[top_5[j]][1])
        regr.fit(max_vals_x.reshape(-1,1), max_vals_y.reshape(-1,1))
        y_pred = regr.predict(max_vals_x.reshape(-1,1))
        mean_x_fr += temp_x/(5*feats)
        ax.text(60, -6, 0.1,  
                "y= %.2f x + %.2f" % (regr.coef_[0,0], regr.intercept_[0]),
                zdir = None,
                color = 'red')
        ax.text(70, -10, 0.1,  
                "r2={0:.3}".format(r2_score(max_vals_y.reshape(-1,1), y_pred)),
                zdir = None,
                color = 'red')
        ax.text(20, -10, 1.3,  
                "L1/Ft: %.2f" % (mean_x_fr/(i+1)),
                zdir = (1, 1, 0.012),
                color = 'green')
        x_lin = range(feats)
        y_lin = [regr.coef_[0,0]*x+regr.intercept_[0] for x in x_lin]
        reg_line = ax.plot(x_lin, y_lin, "r-", linewidth=2)
        line = ax.scatter(max_vals_x, max_vals_y, s=40)
        index_seen.append(i)
        surf = ax.plot_surface(x, y, cv_all, cmap=cm.coolwarm)
        ax.set_title(label)
        ax.set_xlim(2, 77)
        ax.set_ylim(2, 77)
        ax.set_zlim(0,1)
        ax.set_xlabel('Nodes Layer 1')
        ax.set_ylabel('Nodes Layer 2')
        ax.set_zlabel('Mean CV Score')
        plt.savefig(str(i)+".png")
        plt.show()
    
###############################################################################
dir_name = os.path.abspath(os.path.dirname(__file__))

data_file = 'processed_data.csv'
file_path = os.path.join(dir_name, data_file)
data = pandas.read_csv(file_path)
encoder = preprocessing.LabelEncoder() 
encoder.fit(data.iloc[:,-1])
data.iloc[:,-1] = encoder.transform(data.iloc[:,-1])

#Prep the features and labels
features = np.array(data.iloc[:,1:78])
labels = np.array(data.iloc[:,-1])

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

###############################################################################
anim = FuncAnimation(fig, update, frames=range(15), repeat=False)
if len(sys.argv) > 1 and sys.argv[1] == 'save':
    anim.save('surf.gif', dpi=80, writer=ImageMagickFileWriter(fps=1))
else:
    plt.show()