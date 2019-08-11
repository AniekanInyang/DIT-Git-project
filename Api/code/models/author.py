
from db import db

class AuthorModel(db.Model):
  __tablename__ = "authors" 
  id = db.Column(db.Integer, primary_key=True)      
  name = db.Column(db.String(80))
 
  books = db.relationship("BookModel", lazy="dynamic")

  def __init__(self, name):
    self.name = name
   
  
  def json(self):
     
      return {"id": self.id, "name": self.name}
  
  def delete_from_db(self):
  
      db.session.delete(self)
      db.session.commit()


  def save_to_db(self):
    db.session.add(self)
    db.session.commit()

  @classmethod #we introduce this because we rae using the classname in our method which is User as we didn't use self
  def find_by_name(cls, name):
    return cls.query.filter_by(name=name).first()
    

  @classmethod #we introduce this because we rae using the classname in our method which is User as we didn't use self
  def find_by_id(cls, _id):
    return cls.query.filter_by(id=_id).first()
    
 