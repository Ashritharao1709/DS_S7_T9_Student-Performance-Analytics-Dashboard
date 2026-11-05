import matplotlib.pyplot as plt
y1=[]
y2=[]
x=range(-100,100,10)
for i in x:y1.append(i**2)
for i in x:y2.append(-i**2)

plt.plot(x,y1)
plt.plot(x,y2)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-2000,2000)
plt.axhline(0)
plt.axvline(0)
plt.savefig("quad.png")
plt.show()

import matplotlib.pyplot as plt
x_value=[0,1,2,3,4,5]
y_value=[0,1,4,9,16,25]
plt.scatter(x_value, y_value, s=30, color="blue")
plt.show()

import matplotlib.pyplot as plt

x_value = [0,1,2,3,4,5,6,7,8,9]
y_value = [2,4,5,7,6,8,9,11,12,12]

plt.scatter(x_value, y_value, s=30, color="blue", label="star", marker="*")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Scatter Plot Example")

plt.legend()

plt.show()

import matplotlib.pyplot as plt
v=[5,6,3,7,2]
n=["A","B","C","D","E"]
plt.bar(n,v,color="green")
plt.show()

import matplotlib.pyplot as plt
v=[5,6,3,7,2]
n=["A","B","C","D","E"]
plt.barh(n,v,color="green")
plt.show()

import matplotlib.pyplot as plt

# Data
heights = [10, 15, 7, 20, 12]
names = ["A", "B", "C", "D", "E"]
c1 = ["blue","green"]
c2=['b','g']

# Create bar graph
plt.bar(names, heights, width=0.8, color=c1)

# Add labels and title
plt.xlabel("x - axis")        # X-axis label
plt.ylabel("y - axis")            # Y-axis label
plt.title("My Bar Chart")   # Graph title

# Show graph
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

plt.subplot(2,3,1)
plt.plot(df['sepal length (cm)'], df['petal length (cm)'])

plt.subplot(2,3,2)
plt.plot(df['sepal length (cm)'], df['petal length (cm)'])

plt.subplot(2,3,3)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c='green')
plt.title('subplott')

plt.show()

import matplotlib.pyplot as plt

ages = [2,4,6,10,46,37,28,84,63,53,66,22,19,40]

range=(0,100)
bins=10
plt.hist(ages, bins, color='skyblue', histtype='bar',rwidth=0.8)

plt.xlabel("Age")
plt.ylabel("No. of People")
plt.title("My Histogram")

plt.show()

import matplotlib.pyplot as plt

x = [2,4,6,10,46,37,28,84,63,53,66,22,19,40]
plt.hist(x, bins=10, color='green', alpha=0.5)
plt.show()

#implementing seaborn
import seaborn as sns, matplotlib.pyplot as plt
data = sns.load_dataset("iris")
sns.lineplot(x="sepal_length", y="petal_length", data=data)
plt.show()

#customization of seaborn
import seaborn as sns, matplotlib.pyplot as plt
data = sns.load_dataset("iris")

sns.set_style("dark")

sns.lineplot(x="sepal_length", y="petal_length", data=data)
plt.show()

#removal of spines
#customization of seaborn
import seaborn as sns, matplotlib.pyplot as plt
data = sns.load_dataset("iris")
sns.lineplot(x="sepal_length", y="petal_length", data=data)
sns.despine()
plt.show()

#KDE plot visualization with pandas and seaborn
import seaborn as sns
data=sns.load_dataset("iris")
sns.kdeplot(data=data, x="sepal_length")
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
x=np.random.randn(200)
y=np.random.randn(200)
sns.kdeplot(x=x,fill=True)
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
x=np.random.randn(200)
sns.kdeplot(y=x)
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
iris=sns.load_dataset("iris")
setosa=iris.loc[iris.species=="setosa"]
sns.kdeplot(x=setosa.petal_length, y=setosa.petal_width,fill=True)
plt.title("KDE Plot of Setosa Petal Dimensions")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("/content/tips.csv")
sns.countplot(x='sex', hue='smoker', data=df)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("/content/tips.csv")
sns.countplot(x='sex', data=df, palette='Set1')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("/content/tips.csv")
sns.countplot(x='sex', hue='smoker', data=df, color='green')
plt.show()

import matplotlib.pyplot as plt

marks = [10, 21, 12, 30, 23, 26, 17, 100]
plt.boxplot(marks)
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Box Plot of Marks")
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.random.randint(1, 100, (10, 10))
sns.heatmap(data)
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %pip install dash

import dash
from dash import dcc,html
print(dash.__version__)

!pip install dash jupyter-dash

import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dash Tutorial"),
    dcc.Graph(
        id="example",
        figure={
            "data": [
                {
                    "x": [1, 2, 3, 4, 5],
                    "y": [5, 4, 7, 4, 8],
                    "type": "line",
                    "name": "Trucks"
                },
                {
                    "x":[1,2,3,4,5],
                    "y":[6,3,5,3,7],
                    "type":"bar",
                    "name":"Ships"
                }
            ],
"layout":{
    "title":"Basic DashBoard"
}
        }
    )
])
app.run(debug=True, port=8050)

!pip install dash jupyter-dash -q

from dash import Input , Output
id='input'
id='output'

app=dash.Dash(__name__)
app.layout=html.Div([
    html.H1("Square Calculator"),
    dcc.Input(
        id='input',
        type="text",
        placeholder='Enter a number'
    ),
    html.Div(
        id='output'
    )
])
@app.callback(
    Output(component_id='output',component_property='children'),
    Input(component_id='input',component_property='value')
)
def update_value(input_data):
  try:
      return str(float(input_data) **2)
  except:
      return "Error, the inputis not a number"
app.run(debug=True, port=8051)