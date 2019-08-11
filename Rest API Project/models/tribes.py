from db import db


class TribesModel(db.Model):
    __tablename__ = 'tribes'

    tribe = db.Column(db.String(100), primary_key=True)

    food = db.relationship('FoodModel', lazy='dynamic')

    def __init__(self, tribe):
        self.tribe = tribe

    def json(self):
        return {'tribe': self.tribe, 'food': [f.json_minus_tribe() for f in self.food.all()]}

    @classmethod
    def find_by_tribe(cls, name):
        return cls.query.filter_by(tribe=name).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()
