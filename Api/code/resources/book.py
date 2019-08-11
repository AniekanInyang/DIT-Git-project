# import sqlite3
from flask import Flask, request
from flask_restful import Resource, reqparse
from models.book import BookModel

class Book(Resource):
  parser = reqparse.RequestParser()
  parser.add_argument("title", type=str, required=True, help="Name Cannot be empty")
  parser.add_argument("author_name", type=str, required=True, help="Author Name Cannot be empty")
  parser.add_argument("publisher_name", type=str, required=True, help="Publsher Name Cannot be empty")
  parser.add_argument("year", type=int, required=True, help="Please enter the year the book was publish")


  
  def get(self, isbn):
    #fetching item from our database
    book = BookModel.find_by_isbn(isbn)
    if book:
      return book.json()     

    
    return {"message": "Book not found"}, 404


    
  
  
  def post(self, isbn):
    if BookModel.find_by_isbn(isbn):  
      return {"message": "A Book with the number '{}' already exist. ".format(isbn)}, 400
    
    data = Book.parser.parse_args()
   
    book = BookModel(isbn, data["title"], data["publisher_name"], data["author_name"], data["year"])
    
    try:
      book.save_to_db()
    except:
      return {"message": "An error occurred in inserting the book"}, 500 #internal server error
    
    return book.json(), 201

 
  
  
 
class BookList(Resource):
  def get(self):
    return {"books": [book.json() for book in BookModel.query.all()]}

