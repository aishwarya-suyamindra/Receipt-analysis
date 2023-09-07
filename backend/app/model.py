import json

class ProductEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Product):
            return {
                "name": obj.name,
                "price": obj.price
            }
        return super().default(obj)

class ExpenseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Expense):
            return {
                "category": obj.category,
                "tax": obj.tax,
                "total": obj.total,
                "items": [product.toJson() for product in obj.products]
            }
        return super().default(obj)

class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def toJson(self):
        return json.dumps(self, cls=ProductEncoder)

class Expense:
    products = []

    def __init__(self, category, date, tax, total, items):
        self.tax = tax
        self.total = total
        self.category = category
        self.date = date
        for item in items:
            self.products.append(Product(item[0], item[1]))

    def toJson(self):
        return json.dumps(self, cls=ExpenseEncoder)