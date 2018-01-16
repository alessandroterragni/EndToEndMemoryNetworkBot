# EndToEndMemoryNetworkBot
End to end memory networks for task oriented dialog agents on a generated dataset

To run the bot:

• put all the files of the dataset you wanna use in the data/origin folder

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

    python3 main.py --infer.


