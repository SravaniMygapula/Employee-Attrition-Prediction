from flask import * 
import pickle
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# Use pickle to load in the pre-trained model.

filename = 'employee.pkl'
model = pickle.load(open(filename, 'rb'), encoding='latin1')

app = Flask(__name__,)
@app.route('/')
def home():
	return render_template('main.html')

@app.route('/result.html')
def results():
	return render_template('result.html')


@app.route('/main', methods=['GET', 'POST'])
def main():
	if  request.method == 'POST':
		age = request.form['age']
		businesstravel = request.form['businesstravel']
		dailyrate = request.form['dailyrate']
		department = request.form['department']
		distanceformhome = request.form['distanceformhome']
		education = request.form['education']
		educationfield = request.form['educationfield']
		employeecount = request.form['employeecount']
		environmentsatisfaction = request.form['environmentsatisfaction']
		gender = request.form['gender']
		hourlyrate = request.form['hourlyrate']
		jobinvolvement = request.form['jobinvolvement']
		joblevel = request.form['joblevel']
		jobrole = request.form['jobrole']
		jobsatisfaction = request.form['jobsatisfaction']
		maritalstatus = request.form['maritalstatus']
		monthlyrate = request.form['monthlyrate']
		numcompaniesworked = request.form['numcompaniesworked']
		overtime = request.form['overtime']
		percentsalaryhike = request.form['percentsalaryhike']
		performancerating = request.form['performancerating']
		relationshipsatisfaction = request.form['relationshipsatisfaction']
		stockoptionlevel = request.form['stockoptionlevel']
		totalworkingyears = request.form['totalworkingyears']
		trainingtimeslastyear = request.form['trainingtimeslastyear']
		worklifebalance = request.form['worklifebalance']
		yearsatcompany = request.form['yearsatcompany']
		yearsincurrentrole = request.form['yearsincurrentrole']
		yearssincelastpromotion = request.form['yearssincelastpromotion']
		yearswithcurrmanager = request.form['yearswithcurrmanager']
	input_variables = pd.DataFrame([[age, businesstravel, dailyrate, department, distanceformhome, education, educationfield, employeecount, environmentsatisfaction, gender, hourlyrate, jobinvolvement, joblevel, jobrole, jobsatisfaction, maritalstatus, monthlyrate, numcompaniesworked, overtime, percentsalaryhike, performancerating, relationshipsatisfaction, stockoptionlevel, totalworkingyears, trainingtimeslastyear, worklifebalance, yearsatcompany, yearsincurrentrole, yearssincelastpromotion, yearswithcurrmanager ]],
                                       columns=['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
                                       dtype=float)
	prediction = model.predict(input_variables)
	return render_template('result.html',result=prediction)


                                    
dataset=pd.read_csv("C:/Users/SAHITHI/Downloads/webapp/Employee-Attrition.csv")
dataset.drop(["StandardHours"],axis=1, inplace = True)
dataset.drop(["Over18","EmployeeNumber"],axis=1, inplace = True)  

encoder=LabelEncoder()
for col in dataset.columns:
	if dataset[col].dtypes=='object':
		dataset[col]=encoder.fit_transform(dataset[col])

corr = dataset.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
	for j in range(i+1, corr.shape[0]):
		if corr.iloc[i,j] >= 0.9:
			if columns[j]:
				columns[j] = False
selected_columns = dataset.columns[columns]
dataset = dataset[selected_columns]

result = pd.DataFrame()
result['Attrition'] = dataset.iloc[:,1]

dataset.drop(["Attrition"],axis=1, inplace = True)

x_train, x_test, y_train, y_test = train_test_split(dataset.values, result.values, test_size = 0.2)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
acc_log = metrics.accuracy_score(y_pred,y_test)

filename = 'employee.pkl'
pickle.dump(classifier, open(filename, 'wb'))


if __name__ == '__main__':
	app.run(debug=True)