import re
import pandas as pd

df=pd.DataFrame({'Device_Type': ["AXO145", "TRU151","ZOD231","YRT326","LWR245"],
                 'Stats_Access_Link': ["<url>https://xcd32112.smart_meter.com<url>" ,
                                       "<url>http://tXh67.dia_meter.com<url>",
                                       "<url>http://yT5495.smart_meter.com<url>",
                                       "<url>https://ret325_TRu.crown.com<url>",
                                       "<url>https://luwr3243.celcius.com<url>"]})


def url_extract(device_type):
    url_regex = r'(https?:\/\/)?([\da-zA-Z\._]+)\.([a-z\.]{2,6})([\/\w\.]*)*\/?'
    text=str(df.loc[df["Device_Type"]==device_type,"Stats_Access_Link"])
    url = re.search(url_regex,text).group(0)
    return url
print(url_extract("AXO145"))