
from flask_restful import Resource, reqparse
from models.book import BookModel


class Book(Resource):
    parser=reqparse.RequestParser()

    parser.add_argument('author',
                        type=str,
                        required=True,
                        help="This field cannot be empty"
                        )

    parser.add_argument('isbn',
                        type=str,
                        required=True,
                        help="This field cannot be empty"
                        )
    parser.add_argument('publisher',
                        type=str,
                        required=True,
                        help="This field cannot be empty"
                        )
    parser.add_argument('year',
                        type=str,
                        required=True,
                        help="This field cannot be empty"
                        )
    

    def post(self, name):
        if BookModel.find_by_title(name):
            return {"message": "An Author with name '{}' already exist".format(name)},400

        data= Book.parser.parse_args()

        book=BookModel(name, **data)
        try:
            book.save_to_db()
        except:
            return {"message": "An error occurred inserting the item broe"}, 500

        return book.json(),201

    def get(self, name):
        books=BookModel.find_by_title(name)
        if books:
            return books.json()
        return {"message": "Book not found"},404