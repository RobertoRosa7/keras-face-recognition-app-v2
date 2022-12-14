# Import Kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import others dependencies
import os
import cv2
import tensorflow as tf
import numpy as np
from layers import L1Dist


# Build App and Layer
class CamApp(App):

  # Build App
  def build(self):

    # Main Layout Component
    self.web_cam = Image(size_hint=(1, .8))
    self.button = Button(text='Verify', on_press=self.verify, size_hint=(1, .1))
    self.verification_label = Label(text='Verification Uninitiated', size_hint=(1, .1))

    # Add Items to layout
    layout = BoxLayout(orientation='vertical')
    layout.add_widget(self.web_cam)
    layout.add_widget(self.button)
    layout.add_widget(self.verification_label)

    # Load tensorflow/keras model
    self.model = tf.keras.models.load_model('SiameseModel.h5', custom_objects={'L1Dist': L1Dist})

    # Setyp video capture device
    self.capture = cv2.VideoCapture(0)
    Clock.schedule_interval(self.update, 1.0/33.0)

    return layout


  # Run continuosly to get webcam feed
  def update(self, *args):
    # Read fram from opencv
    _, frame = self.capture.read()
    frame = frame[120:120+250, 200:200+250, :]

    # Flip horizontal and convert image to texture
    buf = cv2.flip(frame, 0).tostring()
    img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    self.web_cam.texture = img_texture


  # Load image from file and convert to 100x100px
  def preprocess(self, file_path):
    byte_img =  tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    img = tf.image.resize(img, (100,100))
    img = img / 255.0

    return img

  
  # Bring over verification  function
  def verify(self, *args):
    
    # Specify threshold
    detection_threshold = 0.99
    verification_threshold = 0.88


    # Capture input image from our webcam
    SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
    _, frame = self.capture.read()
    frame = frame[120:120+250, 200:200+250, :]
    cv2.imwrite(SAVE_PATH, frame)

    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold:  Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive prediction / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    # Set verification text
    self.verification_label.text = 'Verified' if verified == True else 'Unverified'
    
    # Log
    Logger.info(f'Detection: {detection}')
    Logger.info(f'Verification: {verification}')
    Logger.info(f'Verified: {verified}\n')

    return results, verified 


if __name__ == '__main__':
  CamApp().run()