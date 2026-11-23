from narwhals import corr
import numpy as np
from sklearn.linear_model import LinearRegression


class HealthAnalyzer:
    """
    Klass som analyserar hälsodata.
    """

    def __init__(self, data):
        self.data = data

    def summary_statistics(self):
        return self.data.describe()

    def correlation_matrix(self):
        return self.data.corr(numeric_only=True)

    def calculate_mean_age(self):
        return np.mean(self.data["age"])

    def blood_pressure_regression(self):

        x = self.data[["age", "weight"]]
        y = self.data["systolic_bp"]

        model = LinearRegression()
        model.fit(x, y)

        return model.score(x, y)

    
    def correlation_heatmap(self):

        import matplotlib.pyplot as plt
        import seaborn as sns

        corr = self.data.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))

        sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )

        plt.title("Correlation Matrix")

        plt.show()


    def plot_age_vs_bp(self):

        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.regplot(
        data=self.data,
        x="age",
        y="systolic_bp"
        )

        plt.title("Age vs Systolic Blood Pressure")
        plt.xlabel("Age")
        plt.ylabel("Systolic Blood Pressure")

        plt.show()