from flask_restful import Resource, reqparse
from models.food import FoodModel


class Food(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('tribe',
                        type=str,
                        required=True,
                        help="tribe field cannot be left blank!"
                        )
    parser.add_argument('ingredients',
                        type=str,
                        required=True,
                        help="ingredients field cannot be left blank!"
                        )
    parser.add_argument('recipe',
                        type=str,
                        required=True,
                        help="recipe field cannot be left blank!"
                        )

    def get(self, name):
        food = FoodModel.find_by_name(name)
        if food:
            return food.json()
        return {'message': 'Food not found'}, 404

    def post(self, name):
        if FoodModel.find_by_name(name):
            return {'message': "An food with name '{}' already exists.".format(name)}, 400

        data = Food.parser.parse_args()

        food = FoodModel(name, data["tribe"], data["ingredients"], data["recipe"])

        try:
            food.save_to_db()
        except:
            return {"message": "An error occurred inserting the item."}, 500

        return food.json(), 201

    def delete(self, name):
        food = FoodModel.find_by_name(name)
        if food:
            food.delete_from_db()
            return {'message': 'Food deleted.'}
        return {'message': 'Food not found.'}, 404

    def put(self, name):
        data = Food.parser.parse_args()

        food = FoodModel.find_by_name(name)

        if food:
            food.tribe = data['tribe']
            food.ingredients = data['ingredients']
            food.recipe = data['recipe']
        else:
            food = FoodModel(name, data["tribe"], data["ingredients"], data["recipe"])

        food.save_to_db()

        return food.json()


class FoodList(Resource):
    def get(self):
        return {'food': list(map(lambda x: x.json(), FoodModel.query.all()))}
