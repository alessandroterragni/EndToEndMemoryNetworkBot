# Using End to End Memory Networks for task oriented dialog systems

Data Mining Seminar

Alessandro Terragni

Supervisor: Mykola Pechenizkiy

To have a complete overview of the project, read the pdf attached.

## Data generator
This code generates a labelled dialogue dataset conforming to the The (6) dialog bAbI tasks database (https://research.fb.com/downloads/babi/), but on a soccer training recommendation app.

You can generate different types of dialogue:

Single intent dialogues: in these dialogues the user states one item per sentence. For example:
1. hi        hello!, what can I help you with today?
2. I want to train on Defense        What is the average age of the training group?
3. Under 11       In which phase do you want to train?
4. Cool-down phase            Who you want to train: a single player, a Group or the whole Team?
5. I would like to train the Group    How many goalkeepers will join the training?
    6 just 0    apicall:  MAIN_FOCUS = Defense  AGE = Under 11
    PHASE = Cool-down  GROUP_SIZE = Group GOAL_KEEPERS = 0

Double intent dialogue: in these dialogues two items can be stated in the same sentence. For example:
1. hi    hello!, what can I help you with today?
2. My Team struggles in Tactics   What is the average age of the training group?
3. Under 9    In which phase do you want to train??
4. During the shot on goal phase   How many goalkeepers will join the training?
5. only 8    apicall:  MAIN_FOCUS = Tactics  AGE = Under 9
    PHASE = shot on goal  GROUP_SIZE = Team GOAL_KEEPERS = 8
 
In this example, the user has stated two objects in one sentence: ”My team struggles in Tactics”, selecting both the group size and the main focus in just one sentence.

You can generate also triple intent, quadruple intent or just mix them in a complete dataset.


## Model
With this code, you can train an End To End Memory Network on the dataset generated with the data generator, and then use it as engine for a chatbot.

To run the bot:

• put all the files of the dataset you wanna use in the model/data/original 
You can generate the dataset on your own using the data generator, or you can use one of the ready to use datasets that you can find in the dataser folder.

• prepare data: 
    
    python3 main.py --prep_data

• train the model:

    python3 main.py --train

• to run the terminal interface::

    python3 main.py --infer 
      
 • to run the Flask web interface
     
      python3 app.py                      

 
Training the model takes time, if you want, you can use one of the model already trained. 
You can find the files in trained model folder.
Just move the content of ckpt and processed in the corresponding folders inside the model folder.
Then just do:

    python3 main.py --infer


## Dependencies
• Python3 • tensorflow • numpy
• scipy
• sklearn
• Flask
• nltk
• six

## Credits
- https://github.com/domluna/memn2n
- https://github.com/voicy-ai/DialogStateTracking
