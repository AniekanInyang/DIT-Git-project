from db import db

class FoodModel(db.Model):
    __tablename__ = 'food'

    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(50))
    ingredients = db.Column(db.String(200))
    recipe = db.Column(db.String(300))

    tribe_id = db.Column(db.Integer, db.ForeignKey('tribes.id'))
    tribe = db.relationship('TribeModel')

    def __init__(self, name, ingredients, recipe, tribe_id):
        self.name = name
        self.ingredients = ingredients
        self.recipe = recipe
        self.tribe_id = tribe_id

    def json(self):
        return {
        'name': self.name,
        'ingredients': self.ingredients,
        'recipe': self.recipe
        }

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name=name).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
