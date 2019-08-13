from flask import request
from flask_restful import Resource
from models.foodmodel import FoodModel

class Food(Resource):
    def get(self, name):
        food = FoodModel.find_by_name(name)
        if food:
            return food.json()
        return {'message': 'Food not found'}


    def post(self, name):
        if FoodModel.find_by_name(name):
            return {'message': "An item with name '{}' already exists.".format(name)}

        data = request.get_json()

        foods = FoodModel(name, **data)

        try:
            FoodModel.save_to_db(foods)
        except:
            return {"message":"An error occurred inserting the item"}
        return foods.json(),201

class Recipe(Resource):
    def get(self, name):
        food = FoodModel.find_by_name(name)
        if food:
            return {'recipe': food.recipe}
        return {'message': 'Food not found'}
