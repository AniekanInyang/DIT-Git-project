import sqlite3
from flask_restful import Resource, reqparse
from models.author import AuthorModel

 
class AuthorList(Resource):
  def get(self):
    return {"publishers": [author.json() for store in AuthorModel.query.all()]}