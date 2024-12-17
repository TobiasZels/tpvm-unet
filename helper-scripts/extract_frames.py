import os
import cv2
import qrcode 

i = 0
for root, dirs, files in os.walk('./frames'):
    if "qr" in os.path.basename(root):
        if(files[-1] == "payload.txt"):
            f = open(root + "/"+ files[-1], "r",errors="ignore")
            payload = f.read()
            print(payload)

            try:
                os.mkdir('./data/' + str(i)) 
            except OSError as error:
                print(error)  
     
            img = qrcode.make(payload)
            img.save('./data/' + str(i) + '/qr.png')
            i += 1
        count = 0
        for file in files:
            if ".webm" in file:
                vid = cv2.VideoCapture(root + "/"+ file)
                success,image = vid.read()
                
                while success:
                    cv2.imwrite('./data/' + str(i) + '/frame' + str(count) + '.jpg' , image)      
                    success,image = vid.read()
                    print('Read a new frame: ', success)
                    count += 1
        print(files)