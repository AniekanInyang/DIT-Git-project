from db import db


class FoodModel(db.Model):
    __tablename__ = 'food'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    ingredients = db.Column(db.String(100))
    recipe = db.Column(db.String(100))

    tribe = db.Column(db.String(100), db.ForeignKey('tribes.tribe'))
    tribes = db.relationship('TribesModel')

    def __init__(self, name, tribe, ingredients, recipe):
        self.name = name
        self.tribe = tribe
        self.ingredients = ingredients
        self.recipe = recipe

    def json(self):
        return {'name': self.name, 'ingredients': self.ingredients, 'recipe': self.recipe, 'tribe': self.tribe}

    def json_minus_tribe(self):
        return {'name': self.name, 'ingredients': self.ingredients, 'recipe': self.recipe}

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name=name).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()
