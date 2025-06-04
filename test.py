from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Merhaba, Flask çalışıyor!'

if __name__ == '__main__':
    app.run(debug=True, port=5000)