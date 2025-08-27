from flask import Flask
from flask import request


app = Flask(__name__)

# ルーティングの基本
@app.route('/')
def index():
    return "Hello, Flask!"

# 複数のルーティングを設定する
@app.route('/about')
def about():
    return "This is the about page."

# パスパラメータ
@app.route('/hello/<username>')
def greet_user(username):
    return f"Hello, {username}!"

# 型指定のパスパラメータ
@app.route('/user/<int:user_id>')
def show_user(user_id):
    return f"User ID is {user_id}"

# クエリパラメータ
@app.route('/search')
def search():
    query = request.args.get('query')
    return f"Search results for: {query}"

if __name__ == '__main__':
    app.run(debug=True)