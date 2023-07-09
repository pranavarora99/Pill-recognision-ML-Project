from flask import Flask, render_template, session, redirect, url_for, request
import sqlite3 as sql
import pandas as pd
import numpy as np
import pickle
import glob
import os
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
# for classifier
import torch, torchvision
# from PIL import Image
from torchvision import transforms
from PIL import Image

img_trans = transforms.Compose([transforms.Resize((128, 128)),
                                 transforms.CenterCrop(100),
#                                 transforms.ToTensor(),
])
app = Flask(__name__,static_folder='static/')
app.config['SECRET_KEY'] = '78sOME098random987key10847'
# def checkFile():
#   path = "G:\Shared drives\CS150Project\CSCI150 files\model data"




app = Flask(__name__,static_folder='static/')
app.config['SECRET_KEY'] = '78sOME098random987key10847'
# def checkFile():
#   path = "G:\Shared drives\CS150Project\CSCI150 files\model data"



def loadSymps():
  X = pd.read_csv('datasets/training_data.csv')
  symptoms = X.columns
  symptoms = symptoms[:len(symptoms)-2]
  return symptoms

# function called on load, removes previous session data:
@app.before_first_request
def clearSesh():
  session.clear()

# login or signup page, this is the root page
@app.route("/")
def index():
  return render_template('login_signupmain.html',methods=['POST'] )

# record a new user signup
@app.route("/index" , methods=['POST','GET'])
def signup():
  if request.method == "POST":
    usr = request.form["firstname"]
    email = request.form["mail"]
    key = request.form["pass"]
    path = "templates\db2.db"

    with sql.connect(path) as con:
      #check if email is already an entry within Database
      cur = con.cursor()
      cur.execute("select email from projects where email=?",(email,))
      emailCheck = cur.fetchall()
      # email already exists, send back to root with message
      if emailCheck:
        return render_template('logredirect.html',error = 1)
      else:
        cur.execute("INSERT INTO projects (name,email,password) VALUES (?,?,?)" ,(usr,email,key) )            
        con.commit()
    # send user to login after signing up
    return render_template('index.html',name = usr,email = email)

  else:
    return render_template('index.html')

@app.route("/log",methods=['POST'])
def login():
  #this grabs user input within form
  if request.method == "POST":
    # usr = request.form["name"]
    email = request.form["email"]
    key = request.form["pass"]
    path = "templates\db2.db"
    #open database connection
    with sql.connect(path) as con:
      cur = con.cursor()
      cur.execute("select password from projects where email=?",(email,))           
      outs = cur.fetchall()
      # if password matches, store user info during runtime
      if outs[0][0] == key:
        session["user"] = email
        return render_template('aftersignin.html',name = email)
      # password fail case
      else:
        return render_template('logredirect.html',error = 2)


@app.route("/Aboutus")
def about_us():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  return render_template('aboutus.html')
@app.route("/contact")
def contact_us():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  return render_template('contact.html')
@app.route("/Account")
def account():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  return render_template('account.html')
@app.route("/home")
def home():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  return render_template('aftersignin.html')

@app.route("/nearby")
def nearby():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  return render_template('gmap.html')
# Present the user with all the symptoms
@app.route("/diagnose")
def diagnose():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3) 
  symptoms = loadSymps()
  symptoms = sorted(symptoms)
  return render_template('try.html',sympList=symptoms)

# The DTM computes based on user input
@app.route("/diagnosis", methods=['POST','GET'])
def diagnosis():
  # check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else: #failure case
    return render_template('logredirect.html',error = 3)
  
  symptoms = loadSymps()
  userSymps = request.form.getlist('symps')
  print(userSymps)
  inputDF = pd.DataFrame(0,index = np.arange(1),columns = symptoms)
  # mark user selections as 1
  for symp in userSymps:
    inputDF[symp] = inputDF[symp].replace(0,1)

  model = pickle.load(open('templates/model.pkl','rb'))
  result = model.predict(inputDF)
  return render_template('prognosis.html',results = result, symps = sorted(userSymps))

@app.route("/upload", methods=['POST','GET'])
def mlInput():
# check log in status, sends to main login page upon failure
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3)
  # If logged in, allow user to input image
  return render_template('mlView.html')

@app.route("/identify",methods=['POST','GET'])
def identify():
  if "user" in session:
    pass
  else:#failure case
    return render_template('logredirect.html',error = 3)

  uploadedFile = request.files['file']
  model = torch.load('templates/mod.pt')
  uploadedFile.save(secure_filename(uploadedFile.filename))
  testimg = Image.open(secure_filename(uploadedFile.filename)).convert('RGB')
  img_to_tensor = transforms.ToTensor() 

  testimg = img_to_tensor(testimg)
  output_batch = torch.stack([testimg])
  output_img = output_batch[0].unsqueeze(0)
  test_output = model(output_img)
  test_sim = test_output[0]

  # search/compare
  path = r"Machine learning model\datasets\CSCI150 files\model data"
  resultmain = 0
  maxi= float(0) 
  for i in range (0, 1153):
    feat_model = torch.load(path+"\\feats" + str(i) + ".pt") 
    result = torch.nn.functional.cosine_similarity(feat_model, test_sim, dim=0)
#     print (result.item())
    if(result.item() >= maxi):
      maxi = result.item()
      print(maxi)
      resultmain = i
      
  # CHECK THE FEATURE VECTORS
  root_dir = r"cropped"

  k = 0
  dataset = pd.read_csv("templates/table.csv")
  predicted = "Hello"
  for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
    if(k==resultmain):
      basename = os.path.basename(filename)
      if maxi < 0.45:
        predicted = "pill not found in dataset accurancy too low. potential match:  " + str(dataset[dataset.rxnavImageFileName == basename].name)
      else:
        if maxi < 0.085:
          predicted = str (dataset[dataset.rxnavImageFileName == basename].name)+ "accurancy " + str(maxi*100)
        else:
          predicted = str(dataset[dataset.rxnavImageFileName == basename].name)
    k += 1
  return render_template('mlOut.html',output = predicted)



if __name__ == "__main__":
  app.run(debug=True)