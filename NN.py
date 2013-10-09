import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy import stats, optimize
import time
import smtplib
from datetime import datetime
from sklearn import cross_validation
import math
import matplotlib.pyplot as plt
#kill those warnings, raging!!
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

#incase: manual zscore function
def zscore(zdata):
    zdata = np.asanyarray(zdata, dtype=np.int)
    return (zdata-zdata.mean())/zdata.std()

#email me the running time 
def email_time(starttime, user, pswd, from_user, to_user):
    endtime = datetime.now() - starttime
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(user,pswd)
    subject = 'Dude, I am done running'
    msg_body = "Check out the running time: "+str(endtime)
    msg = 'Subject: %s\n\n%s' % (subject, msg_body)
    server.sendmail(from_user, to_user, msg)
    server.quit()
    
#text me once done running
def txt_time(starttime, user, pswd, from_user, txt_num):
    endtime = datetime.now() - starttime
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(user,pswd)
    msg = "Running time: "+str(endtime)
    server.sendmail(from_user, txt_num, msg)
    
#converting continuous values into the categorical
def binning_age(age_data):
    age_labels = ['adolescent','young_adult','adult','middle_aged','senior','old']
    age_data = np.asanyarray(age_data)
    #(0-18], (19-25], (26-35], (36-45], (46-60], (61-100]
    bins = [0,18,25,35,45,60,100]
    binned_data = pd.cut(age_data,bins,labels=age_labels)
    return binned_data

#converting hours per week
def binning_hours(hours_data):
    hours_labels = ['part_time', 'full_time', 'over_time', 'workoholic']
    hours_data = np.asanyarray(hours_data)
    #(0-20], (21-40], (41-60], (61-100]
    bins = [0,20,40,60,100]
    binned_hours = pd.cut(hours_data, bins, labels=hours_labels)
    return binned_hours

#converting pay
def bins(cp_data):
    cp_labels = ['low', 'avg','high']
    cp_data = np.asanyarray(cp_data)
    binned_cp = pd.cut(cp_data,3,labels=cp_labels)
    return binned_cp

#confusion matrix to evaluate the accuracy of a classifier
def classifier_accuracy(predicted, actual):
    #confusion list structure:
    #TP | FN
    #FP | TN
    confusion_matrix = [[0, 0], [0, 0]]
    for i in range(len(predicted)):
        if actual[i] == 0:
            if predicted[i] < 0.5:
                confusion_matrix[0][0] += 1#TP
            else:
                confusion_matrix[1][0] += 1#FP
        elif actual[i] == 1:
            if predicted[i] >= 0.5:
                confusion_matrix[1][1] += 1#TN
            else:
                confusion_matrix[0][1] += 1#FN
    accuracy = float(confusion_matrix[0][0] + confusion_matrix[1][1])/sum((map(sum,confusion_matrix))) 
    return accuracy

def sigmoid(X):
    return 1/(1 + math.exp(-X))

def dt_dsigmoid(Y):
    return 1.0 - Y**2

#adapted backpropagation from Neil Schemenauer <nas@arctrix.com>
class NN:
    def __init__(self, ninput, nhidden, noutput):
        #nodes for input, hidden, and output layers
        self.ninput = ninput+1
        self.nhidden = nhidden
        self.noutput = noutput

        self.ainput = self.ninput*[1.]
        self.ahidden = self.nhidden*[1.]
        self.aoutput = self.noutput*[1.]

        self.iweights = (np.random.rand(self.ninput,self.nhidden)-.5).tolist()
        self.oweights = (np.random.rand(self.nhidden, self.noutput)-.5).tolist()

    def forward(self,features):
        #activate inputs 
        for i in range(self.ninput-1):
            self.ainput[i] = features[i]
            
        #activate hidden + bias
        for h in range(self.nhidden):
            val = 0.
            for i in range(self.ninput):
                val += self.ainput[i]*self.iweights[i][h]
            self.ahidden[h] = sigmoid(val)

        for o in range(self.noutput):
            val = 0.
            for h in range(self.nhidden):
                val += self.ahidden[h]*self.oweights[h][o]
            self.aoutput[o] = sigmoid(val)

        return self.aoutput

    def backprop(self,target,lr):
        #error rate for output
        odelta = self.noutput*[0.]
        for i in range(self.noutput):
            error_rate = target[i] - self.aoutput[i]
            odelta[i] = dt_dsigmoid(self.aoutput[i])*error_rate

        #error rate for hidden
        hdelta = self.nhidden*[0.]
        for j in range(self.nhidden):
            error_rate = 0.
            for i in range(self.noutput):
                error_rate = error_rate + odelta[i]*self.oweights[j][i]
            hdelta[j] = dt_dsigmoid(self.ahidden[j])*error_rate
        
        #updating output weights     
        for k in range(self.nhidden):
            for j in range(self.noutput):
                change = odelta[j]*self.ahidden[j]
                self.oweights[k][j] += lr*change

        #updating input weights 
        for i in range(self.ninput):
            for j in range(self.nhidden):
                change = hdelta[j]*self.ainput[i]
                self.iweights[i][j] += lr*change
        #error
        error_rate = 0.
        for i in range(len(target)):
            error_rate += 1/2.*(target[i]-self.aoutput[i])**2

        return error_rate
        
    def train(self, training_set, epoch, lr):
        for i in range(epoch):
            error = []
            for j in training_set:
                features = j[0]
                targets = j[1]
                self.forward(features)
                error.append(self.backprop(targets, lr))
	print len(error)
        
    def predict(self, test_size):
        cmp = []
        for i in test_size:
            cmp.append(self.forward(i[0]))
        return cmp
                
def main():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    adult_data = pd.read_csv('adult.txt', header=None, na_values=' ?')
    colms = ['age', 'workclass',
             'fnlwgt', 'education', 
             'education_num', 'marital_status',
             'occupation', 'relationship',
             'race', 'sex', 'capital_gain',
             'capital_loss', 'hours_per_week', 
             'native_country', 'income']
    adult_data.columns = colms
    key = adult_data['income']
    
    #replacing the NaN's w/the most frequent item in the group
    max_count = lambda x: x.fillna(x.value_counts().index[0])
    #group by <=50K & >50K
    grouped_income = adult_data.groupby(key)
    
    #alredy replaced missing values
    transformed_values = grouped_income.transform(max_count)
    #no extra-space in strings anymore!
    transformed_values[['workclass','education',
                        'marital_status','occupation',
                        'relationship','race', 'sex',
                        'native_country',
                        'income']] = transformed_values[['workclass', 'education',
                                                         'marital_status', 'occupation',
                                                         'relationship','race', 'sex',
                        'native_country','income']].applymap(lambda x: x.strip())
	
    #pre-processing I: discretization - age, hourspweek, cgain, closs
    transformed_values['age'] = binning_age(transformed_values['age'].values)
    transformed_values['hours_per_week'] = binning_hours(transformed_values['hours_per_week'].values)
    transformed_values['capital_loss'] = bins(transformed_values['capital_loss'].values)
    transformed_values['capital_gain'] = bins(transformed_values['capital_gain'].values)

    #pre-processing II: dropping fnlwgt and education_num columns
    transformed_values = transformed_values.drop(['fnlwgt','education_num'], axis=1)
    
    #selecting features/targets and transforming them into binary matrix
    age = pd.crosstab(transformed_values.index, [transformed_values['age']])
    wk = pd.crosstab(transformed_values.index, [transformed_values['workclass']])
    edn = pd.crosstab(transformed_values.index, [transformed_values['education']])
    mrt = pd.crosstab(transformed_values.index, [transformed_values['marital_status']])
    ocp = pd.crosstab(transformed_values.index, [transformed_values['occupation']])
    rln = pd.crosstab(transformed_values.index, [transformed_values['relationship']])
    rce = pd.crosstab(transformed_values.index, [transformed_values['race']])
    sex = pd.crosstab(transformed_values.index, [transformed_values['sex']])
    hpw = pd.crosstab(transformed_values.index, [transformed_values['hours_per_week']])
    nct = pd.crosstab(transformed_values.index, [transformed_values['native_country']])
    gain = pd.crosstab(transformed_values.index, [transformed_values['capital_gain']])
    loss = pd.crosstab(transformed_values.index, [transformed_values['capital_loss']])
    features_binary = pd.concat([age,wk,edn,mrt,ocp,rln,rce,
                                 sex,gain,loss,hpw,nct],axis=1)
    target = DataFrame(transformed_values['income'].values, columns=['trgt'])
    target['trgt'] = target['trgt'].replace('<=50K',float('0'))
    target['trgt'] = target['trgt'].replace('>50K', float('1'))

    #whiten attributes
    features_binary = np.array(features_binary.apply(stats.zscore)).tolist()
    target = np.array(target.values).tolist()
    
    #converting to a desired data type
    input_list = [[[] for i in range(2)] for j in range(len(features_binary))]
    k = 0
    for i in features_binary:
        input_list[k][0] = i
        k+=1
    c = 0
    for j in target:
        input_list[c][1] = j
        c+=1
    input_nodes = len([i for i in features_binary[-1]])
    
    cross_validation = []
    #10-fold-cross-validation 
    for i in range(0,len(input_list)-1, int(len(input_list)*.1)):
        ANN = NN(input_nodes,67, 1)
        train_set = input_list[0:i]+input_list[i+int(len(input_list)*.1):]
        ANN.train(train_set,1,.03)
        test_set = input_list[i:i+int(len(input_list)*.1)]
        predict = ANN.predict(test_set)
        predict = sum(predict, [])
        actual = []
        for i in test_set:
            actual.append(i[1])
        actual = sum(actual,[])
        cross_validation.append(classifier_accuracy(predict,actual))
    print 'The average of accuracy is: ',(sum(cross_validation)/len(cross_validation))*100

if __name__ == '__main__':
        #set the timer
	starttime=datetime.now()
	main()        
	user = '<username>'
	pswd = '<password>'
	from_user = '<username>@gmail.com'
	to_user = '<whatever_email>'
	txt_num = '########@tmomail.net'
    #email or txt once finished running
	#email_time(starttime, user, pswd, from_user, to_user)
    #txt_time(starttime, user, pswd, from_user, txt_num)
