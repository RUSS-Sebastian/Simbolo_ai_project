from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/priceTag')  # The URL route is `/priceTag`
def prices():  # The function name is `prices`
    return render_template('priceTag.html')

@app.route('/teams') 
def teams():  
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)