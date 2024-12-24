import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from CRNN import CRNN
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import PIL
from PIL import Image
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print(pytesseract.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model = CRNN()
#
# # Загрузка весов
# state_dict = torch.load("crnn_synth90k.pt", map_location=device, weights_only=True) # Load model to CPU
#
# # Загрузка весов в модель
# model.load_state_dict(state_dict, strict=False)
# model = model.to(device) # set where to run the model and matrix calculation
# model.eval()

# Preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single mean and std for grayscale

    ]
)


def preprocess(image):
    image = PIL.Image.fromarray(image).convert("L")  # Webcam frames in grey are numpy array format
    # Therefore transform back to PIL image
    #image.show()
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.to(device)
    #image = image.cuda() # for cuda
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    print(image)
    return image  # dimension out of our 3-D vector Tensor


def recognize_text(image):
    # to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # tesseract to extract text
    text = pytesseract.image_to_string(gray, lang="eng")  # ENG
    return text


# Webcam setup
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open the camera.")
    exit()

fps_counter = 0
show_result = "Nothing"
show_score = 0.0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    if fps_counter % 5 == 0:  # Process every 5th frame
        cropped_frame = frame[100:450, 150:570]  # Adjust cropping as needed
        # input_tensor = preprocess(cropped_frame)
        # Text recognise
        text = recognize_text(cropped_frame)

        # with torch.no_grad():
        #     prediction = model(input_tensor)
        # result, score = argmax(prediction)
        # show_result = result
        # show_score = score

    fps_counter += 1
    cv2.putText(frame, f'Text: {text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    print(text)

    # cv2.putText(frame, f'{show_result} (score={show_score:.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (150, 100), (570, 450), (255, 0, 0), 2)  # Visualization of cropped area
    cv2.imshow("Text detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()








