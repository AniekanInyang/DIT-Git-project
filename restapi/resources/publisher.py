from flask_restful import Resource
from models.publisher import PublishersModel


class Publisher(Resource):
    def get(self, name):
        publisher= PublishersModel.find_by_publisher(name)
        if publisher:
            return publisher.json()
        return {"message": "Publisher not found"}, 404

    def post(self, name):
        if PublishersModel.find_by_publisher(name):
            return {"message": "An author with name '{}' already exists.".format(name)},404

        publisher= PublishersModel(name)
        try:
            publisher.save_to_db()
        except:
            return {"message": "An error occurred creating the publisher."},500

        return publisher.json(), 201


