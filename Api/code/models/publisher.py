# import sqlite3
from db import db

class PublisherModel(db.Model):
  __tablename__ = "publishers"

  id = db.Column(db.Integer, primary_key=True)       
  name = db.Column(db.String(80))

  
  books = db.relationship("BookModel", lazy="dynamic")
  #lazy="dynamic" : do not go into items table and create an object for each items

  def __init__(self, name):
    self.name = name
    

  def json(self):
    
    return {"name": self.name, "books": [book.json() for book in self.books.all()]} 

  @classmethod    
  def find_by_name(cls, isbn):
    return cls.query.filter_by(isbn=isbn).first()    
    
  def save_to_db(self):

    db.session.add(self) #The session is a collections of object to be to the database
    db.session.commit() #This whole method is good for both update and insert
  
  def delete_from_db(self):
  
    db.session.delete(self)
    db.session.commit()