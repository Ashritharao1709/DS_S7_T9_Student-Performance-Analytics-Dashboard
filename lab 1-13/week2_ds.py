
import pandas as pd
import numpy as np
df=pd.DataFrame({'Age':[25,30,np.nan,40,50],
   'Department':['HR','Finance','Finance', np.nan,'IT']})
print("original dataset(with missing values):")
print(df)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Department']=df['Department'].fillna(df['Department'].mode()[0])
print(df)

import pandas as pd
import numpy as np
df=pd.DataFrame({'Age':[25,30,np.nan,40,50],
   'Department':['HR','Finance','Finance', np.nan,'IT']})
print("original dataset:")
print(df)

df_ffill=df.copy()
df_ffill.ffill(inplace=True)
print(df_ffill)

df_bfill=df.copy()
df_bfill.bfill(inplace=True)
print(df_bfill)


import pandas as pd
import numpy as np

df=pd.DataFrame({
   'Age':[25,30,np.nan,40,50],
   'Department':['HR','Finance','IT','Finance','IT']
})
print("original dataset:")
print(df)

df_drop_rows=df.dropna()
print("After dropping rows:\n",df_drop_rows)

df_drop_cols=df.dropna(axis=1)
print("After dropping columns:\n",df_drop_cols)


import pandas as pd
df = pd.DataFrame({
    'ID':[1,2,2,3,4,4],
    'Name':['Alice','Bob','Bob','charlie','David','David'],
    'Age':[25,30,30,35,40,40]
})

print("originaldata:\n",df)

df_exact=df.drop_duplicates()
print("\nAfter Exact Match Removal:\n",df_exact)

#3.b Removing Duplicates (Subset-Based Removal)
import pandas as pd

df=pd.DataFrame({
    'ID':[1,2,2,3,4,4],
    'Name':['Alice','Bob','Bob','charlie','David','David'],
    'Age':[25,30,30,35,40,40]
})

print("originaldata:\n",df)

df_subset_id=df.drop_duplicates(subset=['ID'])
print('\nAfter subset-based removal(ID):\n',df_subset_id)

df_subset_name=df.drop_duplicates(subset=['Name'])
print('\nAfter subset-based removal(Name):\n',df_subset_name)

#3.(b)Removing Duplicates (Corresponding Inconsistent Formats)
#3.b.a Data Standardizaation (IOS 8601 Format)
import pandas as pd

df=pd.DataFrame({
    'Date':['2025-01-05','05/03/5789','jan 5,2025','2025.01.03']
})
print("Original Data:\n",df)

df['Date']=pd.to_datetime(df['Date'],errors='coerce').dt.strftime('%y-%m-%d')

print(df);


df=pd.DataFrame({
    'Name':['Alice','BOB','charlie','DAVID']
})
print("original data:\n",df)

df['Name_lower']=df['Name'].str.lower()

df['Name_upper']=df['Name'].str.upper()

print("\n",df)