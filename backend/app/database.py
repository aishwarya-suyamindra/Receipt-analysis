import pymongo as pm

# Connection settings
mongodb_url = "mongodb://localhost:27017/"
database_name = "expensesDB"

# Connect
client = pm.MongoClient(mongodb_url)
database = client[database_name]

class Database:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Database, cls).__new__(cls)
        return cls.instance
    
    def save(self, model):
        expense_document = {
            "category": model.category,
            "total": model.total,
            "tax": model.tax,
            "date": model.date,
            "items": [vars(product) for product in model.products]
        }
        # save the document to the database
        collection = database["Expenses"]
        collection.insert_one(expense_document)