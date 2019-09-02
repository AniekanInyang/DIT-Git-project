from db import db

class AuthorModel(db.Model):
    __tablename__ = 'authors'

    author= db.Column(db.String(120), primary_key=True)
    books=db.relationship('BookModel', lazy='dynamic')

    def __init__(self, author):
        self.author=author

    def json(self):
        return {
            'author': self.author,
            'books':[a.json_less_one() for a in self.books.all()]
        }

    @classmethod
    def find_by_author(cls, author):
        return cls.query.filter_by(author=author).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
