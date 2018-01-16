from flask import Flask
from flask import render_template, request
import main as botmodule


app = Flask(__name__)
bot = None
posts = None

@app.route('/')
def index():
    return render_template('index.html')
                
 

@app.route('/chat',methods = ['POST', 'GET'])
def chat():
    if request.method == 'GET':
        msg = str(request.args.get('message'))
        reply = bot.reply(msg)
        memorize(msg,reply)
        if msg == "exit": 
            clear_posts()
            bot.reply("clear")
            render_template("index.html")
        return render_template('index.html', posts = posts)
        
                
            
            
            
def memorize(msg,reply):
    global posts
    posts += [{"message": msg, "answer": reply}]

def clear_posts():
    global posts
    posts = []

    
if __name__ == '__main__':
    # before starting the app, save the inference model
    bot = botmodule.main(['--ui', '--task_id=1'])
    posts = []
    # start app
    app.run(port = 5000)