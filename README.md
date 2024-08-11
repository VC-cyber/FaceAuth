This code uses a HaarCascade classifier for facial detection and passes face information to the authenticator, which is trained by me on celebA and myself. 

To use yourself: 

1. upload your data in jpeg or png format under data under 1, replacing whatever is there currently
2. go through all the files and replace paths of the file paths of the ones you have organized, which is probably similar
3. run augmentation.py to add more augmented data 
4. run preprocessing.py to obtain all the facial information from the haarclassifier of all the data and label it accordingly (0 for not you, 1 for you)
5. run train.py
6. run authenticate.py and it will open a screen with your face and bounding boxes and whether your face is authenticated or not, and
change the confidence rating to however flexible you want the authenticator to be 


NEXT STEPS: 
1. figure out if multiuser has good confidence
2. try out different models for face detection and recognition
3. try to create a script to do all of the process of adding a user/authenticating yourself
