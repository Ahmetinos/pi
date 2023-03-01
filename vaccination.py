#Libraries
import pandas as pd

df=pd.read_csv("Datasets/country_vaccination_stats.csv")
df.head()

#Q4
df["daily_vaccinations"].isnull().sum()
min_vac=df.groupby("country").agg({'daily_vaccinations':"min"}).fillna(0)
for country in min_vac.index:
    df.loc[df["country"]==country,"daily_vaccinations"]=\
        df["daily_vaccinations"].fillna(min_vac.loc[country,"daily_vaccinations"],axis=0)

print(df["daily_vaccinations"].isnull().sum())
print(df[df["country"]=="Kuwait"])

#Q5
highest_median=df.groupby("country").agg({'daily_vaccinations':"median"}).\
    sort_values("daily_vaccinations",ascending=False).head(3).reset_index()["country"]
print("Looking at the median values, the top 3 countries are as follows:{},{} and {}".
      format(highest_median[0],highest_median[1],highest_median[2]))

#Q6
print("Total vaccination(1/6/2021):{}".format(df.loc[df["date"]=="1/6/2021","daily_vaccinations"].sum()))
