
# coding: utf-8

# In[ ]:


# producer.py
import codecs
import time
import json
from kafka import KafkaProducer, KafkaClient

topic='topic0302'

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

e=open("flights.csv",'r').readlines()

flights_clean_all=[]
flights_clean=[]

#on average 14k flights a DAY !!!
#bullshit more like 42k a day, \
#source: https://www.faa.gov/air_traffic/by_the_numbers/

nlines=500 #could be defined by the user

times= len(e) // nlines

rest = len(e) % nlines

def funct(j,k):
    """creating and sending a batch to KafkaTopic"""
        for i in range(j,k):#len(e)
                columns_first=e[i].split(",")[0:12]
                columns_last=e[i].split(",")[21:25]
                columns=columns_first + columns_last
                producer.send(topic,json.dumps(columns))
                print(columns)

for i in range(1, 200): #times
    j=(i-1)*nlines      
    k=i*nlines
    print("starting row is",j)
    print("last row from the stream is",k-1)
    print("Start streaming in 10 seconds\n")
    time.sleep(5)
    funct(j,k)
    print("The batch is completed!\n")
    time.sleep(10)

