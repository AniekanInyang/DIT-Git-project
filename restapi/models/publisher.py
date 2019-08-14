
from db import db

class PublishersModel(db.Model):
    __tablename__ = 'publishers'

    publisher= db.Column(db.String(120), primary_key=True)

    books=db.relationship('BookModel', lazy='dynamic')

    def __init__(self, publisher):
        self.publisher=publisher

    def json(self):
        return {
            'publisher': self.publisher,
            'books':[a.json() for a in self.books.all()]
        }

    @classmethod
    def find_by_publisher(cls, publisher):
        return cls.query.filter_by(publisher=publisher).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
