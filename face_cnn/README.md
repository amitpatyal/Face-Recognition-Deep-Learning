
**Introduction :-**

	In project, i am going to perform face recognition with OpenCV, Python, and deep learning.

**Prerequisites :-** 

	pip install os, cv2, dlib, openface, face_recognition, pickle, numpy, pandas

**Files Details :-**

	Now we are going to build the face recognition using deep learning but first,
	let us see the file structure and the type of files we will be creating.
	
	public_dataset:- This folder contains images of the user with there sub folder.
	                 Which need to br train. Exemple public_dataset/abcuser


	train_align_dataset:- This folder contains process images of the user with
	                      their sub folder. This will done by script.

	face_id_name.csv:- CSV file contains the user name.
	publicFaceData.pkl :- This is a pickle file in which we store the face tensor.
	helper_function :- This is the Python script in which have helper function.
	
**Here are the 5 steps to create a face recognition with deep learning in Python from scratch :-**

	Import and load the data user face into public dataset folder with subfolders.
	Preprocess user face data.
	Create training face data 
	Build the face model.
	Predict the response	
