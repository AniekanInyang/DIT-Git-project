from flask_restful import Resource
from models.publisher import PublisherModel

class PublisherList(Resource):
  def get(self):
    return {"publishers": [publisher.json() for store in PublisherModel.query.all()]}

# class Publisher(Resource):
#   def get(self, name):
#     publisher = PublisherModel.find_by_name(name)
#     if publisher:
#       return publisher.json()
#     return {"message": "store not found"}, 404


#   def post(self, isbn):
#     if PublisherModel.find_by_isbn(isbn):
#       return {"message": "A store with name '{}' already exist.".format(isbn)}, 400
    
#     publisher = PublisherModel(isbn)
#     try:
#       publisher.save_to_db()
#     except:
#       return {"message": "An error occured while creating store"}, 500
#     return publisher.json(), 201


# class PublisherList(Resource):
#   def get(self):
#     return {"publishers": [publisher.json() for store in PublisherModel.query.all()]}