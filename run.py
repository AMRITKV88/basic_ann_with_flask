from flask import Flask
from first_ann_try import run_ann

app = Flask(__name__)

@app.route("/")
def index():

    test_accuracy = run_ann()
    print("\n\n --> Testing accuracy : ", test_accuracy, "\n\n")
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)
