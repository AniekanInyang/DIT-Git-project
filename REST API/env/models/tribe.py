from db import db

class TribeModel(db.Model):
    __tablename__ = 'tribes'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

    foods = db.relationship('FoodModel', lazy='dynamic')

    def __init__(self, name):
        self.name = name

    def json(self):
        return {
        'name': self.name,'food':[meal.json() for item in self.meals.all()]
        }

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name=name).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
