import sqlite3

connection = sqlite3.connect("data.db")

cursor = connection.cursor()

create_table = "CREATE TABLE users (id int, username text, password text)"
cursor.execute(create_table)

user = (1, "nath", "qwerty")

insert_query = "INSERT INTO users VALUES (?, ?, ?)"
cursor.execute(insert_query, user)

#to inset multiple users

users = [
  (2, "rolf", "qwerty"),
  (3, "phoenix", "abcdef")
]
cursor.executemany(insert_query, users)

select_query = "SELECT * FROM users"
for row in cursor.execute(select_query):
  print(row)


connection.commit() #to add changes

connection.close() # to close connection