
import cv2
import numpy as np
import math


def show_image(image):
    cv2.imshow('', image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()



def run_filter_on_camera(image_filter):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        result_image = image_filter(frame)

        # Display the resulting frame
        cv2.imshow('frame', result_image)

        # We break out of the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def run_filter_on_camera_withparams(image_filter):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        result_image = image_filter(frame)

        # Display the resulting frame
        cv2.imshow('frame', result_image)

        # We break out of the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# In[7]:

font                   = cv2.FONT_HERSHEY_SIMPLEX
pos = (25,100)
fontScale              = 2
fontColor              = (0,255,0)
lineType               = 2


def cv2flip(img, key=0):
    return cv2.flip(img, key)

sepiakernel=np.array([[0.272,0.534,0.131],
                      [0.0349, 0.686,0.168],
                      [0.393,0.769,0.189]])
#with numpy
def imagemirror(img):
    width = img.shape[1]
    img2=img.copy()
    img2[:, :width // 2,:]=img[:, width // 2:,:][:,::-1,:]
    return img2
def grayfilter(img):
    grayimg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grayimg, cv2.COLOR_GRAY2RGB)

def invfilter(img):
    return cv2.bitwise_not(img)

def gaussianblur(img):
    return cv2.GaussianBlur(img, (5,5),0)
def cannyfilter(img):
    cannyimg= cv2.Canny(img, 100,100)
    return cv2.cvtColor(cannyimg, cv2.COLOR_GRAY2RGB)

def sepiafilter(img):
    return cv2.filter2D(img,-1, sepiakernel)

def addtext(image, title, pos):
    cv2.putText(image,title,
    pos,
    font,
    fontScale,
    fontColor,
    lineType)
    return image


def make_grid(image):
    img1=image
    row1=np.concatenate((grayfilter(img1), gaussianblur(img1), invfilter(img1)), axis=1)
    row2=np.concatenate((cannyfilter(img1), img1, sepiafilter(img1)), axis=1)
    row3=np.concatenate((imagemirror(img1), cv2flip(img1, key=1), cv2flip(img1, key=0)), axis=1)
    combined=np.concatenate((row1, row2, row3), axis=0)
#     h,w,_=image.shape

#     for i in range(3):
#         for j in range(3):
#             addtext(combined,f"filter {i} {j}", (25+i*w,100+j*h))
    return combined

def image_resize(image, per=10): 
    dim_final=(int(0.01*per*image.shape[0]), int(0.01*per*image.shape[1]))
    return cv2.resize(image, dim_final)

def rotate_image(image, angle=5):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result

def image_tweak(image, per, angle):
    img1=image_resize(image, per)
    imgout=img1
    if np.abs(angle)>20:
        imgout=rotate_image(img1, angle)
    return imgout
    



def black_and_white(image):
    return_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return_image[:, :, 0] = gray
    return_image[:, :, 1] = gray
    return_image[:, :, 2] = gray
    
    return return_image



import mediapipe as mp

drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles




def run_filter_with_mediapipe_model(mediapipe_model, mediapipe_based_filter):
    cap = cv2.VideoCapture(0)
    
    with mediapipe_model as model:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'.

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            results = model.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result_image = mediapipe_based_filter(image, results)

            cv2.imshow('MediaPipe', result_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()





Holistic = mp.solutions.holistic.Holistic


def draw_holistic_results(image, results, show_hands=True, show_face=True, show_pose=False):
    imgH, imgW, imgC = image.shape  # height, width, channel for image
    # xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
    if show_hands:
        # drawing_utils.draw_landmarks(
        #     image,
        #     results.right_hand_landmarks,
        #     mp.solutions.holistic.HAND_CONNECTIONS#,
        #     # connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
        # )
        if results.left_hand_landmarks:
            mlist=[x for x in results.left_hand_landmarks.landmark]
            x1, y1 = int(mlist[4].x* imgW), int(mlist[4].y* imgH)
            x2, y2 = int(mlist[8].x* imgW), int(mlist[8].y* imgH)
            length = math.hypot(x2-x1, y2-y1)
            angle=np.arcsin((x2-x1)/length)*180/np.pi
                
     
            cv2.circle(image, (x1, y1), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 5, (255, 255, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            # image=image_resize(image, per=100*length/200)
            #image=image_tweak(image, per=100*length/100, angle=angle)
            case1 = angle > 10 and angle < 40
            case2 = angle > 40 and angle < 70
            case3 = angle > 70 and angle < 100
            case4 = angle > 100 and angle < 130
            case5 = angle > 130 and angle < 170
            case6= angle<-10 and angle >-90
            
            if case1:  image=grayfilter(image)
            elif case2: image=sepiafilter(image)
            elif case3: image=invfilter(image)
            elif case4: image= cannyfilter(image)
            # elif case6: image=image_tweak(image, per=100*length/100, angle=angle)
            else: image=cv2.resize(make_grid(image), (imgH, imgW))
            
        if results.right_hand_landmarks:
            mlist=[x for x in results.right_hand_landmarks.landmark]
            x1, y1 = int(mlist[4].x* imgW), int(mlist[4].y* imgH)
            x2, y2 = int(mlist[8].x* imgW), int(mlist[8].y* imgH)
            length = math.hypot(x2-x1, y2-y1)
      
            angle=np.arcsin((x1-x2)/length)*180/np.pi
                
     
            cv2.circle(image, (x1, y1), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 5, (255, 255, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            image=image_tweak(image, per=100+length/10, angle=angle)
          


    
    return image


# In[ ]:


run_filter_with_mediapipe_model(
    mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
    mediapipe_based_filter=draw_holistic_results
)
