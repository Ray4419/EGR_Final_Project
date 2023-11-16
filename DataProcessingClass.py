import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class MPG:
    def __init__(self, file_path='vehicles-processed.csv', columns_to_drop=['make', 'model', 'VClass','gears']):
        self.raw_data = pd.read_csv(file_path)
        self.clean_data = self.drop_columns(columns_to_drop)

        self.modified_data = {}

        self.mpg_col = 'comb08'

        self.trained_models = {}

    def drop_columns(self, columns_to_drop):
        tmpDF = self.raw_data.drop(columns=columns_to_drop, errors='ignore')
        return tmpDF

    def get_year_range(self, start_year, end_year=None):
        if end_year == None:
            end_year = start_year + 10
        result_data = self.clean_data[
            (self.clean_data['year'] >= start_year) & (self.clean_data['year'] <= end_year)].copy()
        # Update modified_data dictionary
        decade_key = f'{start_year}-{end_year}'
        self.modified_data[decade_key] = result_data

        return result_data

    def train_linear_regression(self, train_data, test_data):

        Y_Train = np.array(train_data[self.mpg_col])
        X_Train = np.array(train_data.drop(columns=[self.mpg_col]))

        Y_Test = np.array(test_data[self.mpg_col])
        X_Test = np.array(test_data.drop(columns=[self.mpg_col]))
        #print(type(Y_Train))
        reg = LinearRegression().fit(X_Train, Y_Train)
        Y_predict = reg.predict(X_Test)


        mse = mean_squared_error(Y_Test, Y_predict)
        train_data_df = train_data.drop(columns=[self.mpg_col])
        #print(train_data[self.mpg_col])

        Titles = train_data_df.columns.tolist()
        Coefficient = pd.DataFrame(reg.coef_).T
        Coefficient.columns = Titles
        print(Coefficient)
        return mse



# Example usage:
mpg = MPG()


mse = []

years = np.arange(1985,2015,1)

for d in years:
    mse.append(mpg.train_linear_regression(mpg.get_year_range(d),mpg.get_year_range(d+10)))


#for i in range(len(years)):
#    print(f"MSE trained on {years[i]}: {mse[i]}")

# Plotting
plt.plot(years, mse, marker='o')
plt.title('MSE vs Year')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Print MSE values
for i in range(len(years)):
    print(f"MSE trained on {years[i]}: {mse[i]}")
#print(eighties.head)
# Display the modified DataFrame
#print(seventies)






