from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team')
def teams():
    return render_template('team.html')


@app.route('/essay')
def Essay():
    return render_template('Essay.html')


@app.route('/flashcard')
def FlashCard():
    return render_template('FlashCard.html')

if __name__ == '__main__':
    app.run(debug=True)
