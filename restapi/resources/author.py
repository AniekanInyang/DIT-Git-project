from flask_restful import Resource
from models.author import AuthorModel


class Author(Resource):
    def get(self, name):
        author= AuthorModel.find_by_author(name)
        if author:
            return author.json()
        return {"message": "Author not found"}, 404


    def post(self, name):
        if AuthorModel.find_by_author(name):
            return {"message": "An author with name '{}' already exists.".format(name)},404

        author= AuthorModel(name)
        try:
            author.save_to_db()
        except:
            return {"message": "An error occurred creating the author."},500

        return author.json(), 201


