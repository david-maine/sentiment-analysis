#%%
from pymongo import MongoClient


#%%
client = MongoClient()
db = client.test

#%% [markdown]
# Insert some dummy collections

#%%
person = {
    "name" : "david",
    "age" : "21"
}

people = db.people
person_id = people.insert_one(person).inserted_id
print(person_id)
#%%
