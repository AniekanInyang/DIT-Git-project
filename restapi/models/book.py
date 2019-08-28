from db import db

class BookModel(db.Model):
    __tablename__ = 'books'


    title = db.Column(db.String(120), primary_key=True)
    isbn = db.Column(db.Integer)
    author = db.Column(db.String(120),db.ForeignKey('authors.author'))
    publisher = db.Column(db.String(120), db.ForeignKey('publishers.publisher'))
    year = db.Column(db.String(120))

    publishers=db.relationship('PublishersModel')
    authors=db.relationship('AuthorModel')


    def __init__(self, title, isbn, author, publisher, year):
        self.title=title
        self.isbn=isbn
        self.author=author
        self.publisher=publisher
        self.year=year

    def json(self):
        return {
            'title':self.title,
            'isbn':self.isbn,
            'author':self.author,
            'publisher':self.publisher,
            'year':self.year
        }

    def json_less_one(self):
        return {'title': self.title,
                'isbn': self.isbn,
                'author': self.author,
                'year': self.year
                }


    @classmethod
    def find_by_title(cls, title):
        return cls.query.filter_by(title=title).first()

    @classmethod
    def find_all(cls):
        return cls.query.all()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
