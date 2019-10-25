from flask import Flask, render_template

app = Flask(__name__)

# Set up app.route decorator so runs webpage() function when user loads http://127.0.0.1:5000/
@app.route('/')
def webpage():
    # Render the HTML page Webpage.html from the templates folder
    return render_template('/Webpage.html')

if __name__ == '__main__':
    # Run the application 
    app.run()