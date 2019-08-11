import sqlite3
from db import db

class BookModel(db.Model):
  __tablename__ = "books"

  id = db.Column(db.Integer, primary_key=True)      
  isbn = db.Column(db.Integer)
  title = db.Column(db.String(80))
  
  author_name = db.Column(db.String(80), db.ForeignKey("authors.name"))
  author = db.relationship("AuthorModel")

  publisher_name = db.Column(db.String(80), db.ForeignKey("publishers.name"))
  publisher = db.relationship("PublisherModel")

  year = db.Column(db.Integer)

  
  def __init__(self, isbn, title, author_name, publisher_name, year):
    self.isbn = isbn
    self.title = title
    self.author_name = author_name
    self.publisher_name = publisher_name
    self.year = year

  def json(self):
      return {"isbn": self.isbn, "title": self.title, "author_name": self.author_name, "publisher_name": self.publisher_name, "year": self.year}

  @classmethod        
  def find_by_isbn(cls, isbn):
    return cls.query.filter_by(isbn=isbn).first()   
    
  def save_to_db(self):
  
    db.session.add(self) 
    db.session.commit() 
  def delete_from_db(self):
  
    db.session.delete(self)
    db.session.commit()