from flask import Flask, request

app = Flask(__name__)

@app.route('/runWeb.py', methods=['POST'])
def handle_form():
    data = request.form['data']
    # Do some processing with the data
    return 'Processed data: ' + data

if __name__ == '__main__':
    app.run()