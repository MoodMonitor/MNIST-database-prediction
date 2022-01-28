from functools import partial
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.screenmanager import NoTransition
from kivy.properties import NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from time import time
import cv2 as cv
from kivy.uix.textinput import TextInput
import numpy as np
from keras.models import load_model
import os

Builder.load_string("""
<MainScreen>:
    box_layout: box_layout
    BoxLayout:
        orientation: "vertical"
        pos_hint: {"top": 1, "left": 0.5}
        size_hint: 1, None
        height: self.minimum_height
        PaintWidget:
            id: paint
            size_hint: 1, None
            height: app.height/2.47
            canvas.after:
                Color:
                    rgba: 1,1,1, 0.5
                Line:
                    width: 1.5
                    rectangle: (self.x, self.y , self.width, self.height)
            canvas.before:
                Color:
                    rgba: 0,0,0,1
                Rectangle:
                    pos: self.pos
                    size: self.size  
        BoxLayout: 
            orientation: "horizontal"
            size_hint: 1, None
            height: self.minimum_height
            BoxLayout:
                orientation: "vertical"
                size_hint: None, None
                height: self.minimum_height
                width: self.minimum_width
                canvas.before:
                    Color:
                        rgba: 1,1,1, 0.6
                    Line:
                        width: 1.5
                        rectangle: (self.x, self.y , self.width, self.height)
                Label:
                    size_hint: 1, None
                    height: app.height/10
                    text: "Cropped Image"
                    font_size: 20
                    canvas.before:
                        Color:
                            rgba: 1,1,1, 0.6
                        Line:
                            width: 1.5
                            rectangle: (self.x, self.y , self.width, self.height)
                Image:
                    id: frame
                    alpha: 0
                    color: 0.5, 0.5, 0.5, 0
                    size_hint: None, None
                    width: app.width/2
                    height: app.height - app.height/2.47 - app.height/10 - app.height/10
                    canvas:
                        Color:
                            rgba: 1,1,1, self.alpha
                        Rectangle:
                            texture: self.texture
                            pos: self.pos
                            size: self.size 
            BoxLayout:
                id: box_layout
                orientation: "vertical"
                size_hint: 1, 1
                BoxLayout:
                    orientation: "horizontal"
                    size_hint: 1, 1
                    
                    Button:
                        pos_hint: {"top": 1}
                        size_hint: 1, None
                        height: self.parent.parent.height/3.5
                        text: "Predict"
                        on_release: root.make_a_prediction(self, paint, frame)
                    Button:
                        pos_hint: {"top": 1}
                        size_hint: 1, None
                        height: self.parent.parent.height/3.5
                        text: "Clear"
                        on_release: app.clear_canvas(paint, frame)
    BoxLayout:
        orientation: "horizontal"
        size_hint: 1, None
        height: self.minimum_height
        pos_hint: {"top": 0.1}
        Button:
            size_hint: 1, None
            height: app.height/10
            text: "Prediction Screen"
            on_release: root.manager.current = "Main"
        Button:
            size_hint: 1, None
            height: app.height/10
            text: "Save Screen"
            on_release: root.manager.current = "Train"

<SaveScreen>:
    box_layout: box_layout
    BoxLayout:
        orientation: "vertical"
        pos_hint: {"top": 1, "left": 0.5}
        size_hint: 1, None
        height: self.minimum_height
        PaintWidget:
            id: paint
            size_hint: 1, None
            height: app.height/2.47
            canvas.after:
                Color:
                    rgba: 1,1,1, 0.5
                Line:
                    width: 1.5
                    rectangle: (self.x, self.y , self.width, self.height)
            canvas.before:
                Color:
                    rgba: 0,0,0,1
                Rectangle:
                    pos: self.pos
                    size: self.size  
        BoxLayout: 
            orientation: "horizontal"
            size_hint: 1, None
            height: self.minimum_height
            BoxLayout:
                orientation: "vertical"
                size_hint: None, None
                height: self.minimum_height
                width: self.minimum_width
                canvas.before:
                    Color:
                        rgba: 1,1,1, 0.6
                    Line:
                        width: 1.5
                        rectangle: (self.x, self.y , self.width, self.height)
                Label:
                    size_hint: 1, None
                    height: app.height/10
                    text: "Cropped Image"
                    font_size: 20
                    canvas.before:
                        Color:
                            rgba: 1,1,1, 0.6
                        Line:
                            width: 1.5
                            rectangle: (self.x, self.y , self.width, self.height)
                Image:
                    id: frame
                    alpha: 0
                    color: 0.5, 0.5, 0.5, 0
                    size_hint: None, None
                    width: app.width/2
                    height: app.height - app.height/2.47 - app.height/10 - app.height/10
                    canvas:
                        Color:
                            rgba: 1,1,1, self.alpha
                        Rectangle:
                            texture: self.texture
                            pos: self.pos
                            size: self.size 
            BoxLayout:
                id: box_layout
                orientation: "vertical"
                size_hint: 1, 1
                BoxLayout:
                    orientation: "horizontal"
                    size_hint: 1, 1
                    
                    Button:
                        pos_hint: {"top": 1}
                        size_hint: 1, None
                        height: self.parent.parent.height/3.5
                        text: "Save"
                        on_release: root.save_cropped_image(self, save, paint, frame)
                    Button:
                        pos_hint: {"top": 1}
                        size_hint: 1, None
                        height: self.parent.parent.height/3.5
                        text: "Clear"
                        on_release: app.clear_canvas(paint, frame)
                TextInput:
                    id: save
                    size_hint: 1, None
                    height: app.height/10
                    hint_text: "Enter a Digit that you are will paint"

    BoxLayout:
        orientation: "horizontal"
        size_hint: 1, None
        height: self.minimum_height
        pos_hint: {"top": 0.1}
        Button:
            size_hint: 1, None
            height: app.height/10
            text: "Prediction Screen"
            on_release: root.manager.current = "Main"
        Button:
            size_hint: 1, None
            height: app.height/10
            text: "Save Screen"
            on_release: root.manager.current = "Train"

""")

global MODEL_NAME, X_PADDING, Y_PADDING
MODEL_NAME = "./DigitsSquaredModel_grey.h5"
X_PADDING = 1.83
Y_PADDING = 4.38

def crop_image(painter):
    """
    Crop the image
    :param painter: painter object
    :return: cropped image
    """
    file_name = "temporary_img.png"
    painter.export_to_png(f"./{file_name}")
    image = cv.imread(file_name, cv.COLOR_BGR2GRAY)
    Y, X = np.where(np.all(image == [255, 255, 255], axis=2))
    X, Y = zip(*sorted(zip(X, Y)))
    for index in range(len(X) - 1):
        if X[index + 1] - X[index] > 1:
            break
    crop_img = image[int(min(Y[:index + 1])):int(max(Y[:index + 1])),
               int(min(X[:index + 1])):int(max(X[:index + 1]))]
    crop_img = cv.copyMakeBorder(crop_img, top=int(crop_img.shape[0] / Y_PADDING),
                                 bottom=int(crop_img.shape[0] / Y_PADDING),
                                 left=int(crop_img.shape[0] / X_PADDING) + int(crop_img.shape[0] / Y_PADDING),
                                 right=int(crop_img.shape[0] / X_PADDING) + int(crop_img.shape[0] / Y_PADDING),
                                 borderType=cv.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    cv.imwrite(file_name, crop_img)
    return crop_img


class MainScreen(Screen):
    box_layout = ObjectProperty()

    def make_a_prediction(self, obj, painter, frame):
        """
        Predict painted digit
        :param obj: button object
        :param painter: painter object
        :param frame: Image object
        :return: None
        """
        self._clear_widgets()
        image = crop_image(painter)
        frame.alpha = 1
        frame.source = "temporary_img.png"
        frame.reload()
        image.astype(np.float32)
        image = cv.resize(image, (28, 28))
        image = image / 255
        image = image.reshape(1, 28, 28, 1)
        global MODEL
        label = MODEL.predict(image)
        self.build_prediction_decision(np.argmax(label), label[0][np.argmax(label)], painter, frame)

    def build_prediction_decision(self, predicted_number, accuracy, painter, frame):
        """
        Build a decision buttons if prediction was good or bad
        :param predicted_number: predicted number by model
        :param accuracy: accuracy of prediction
        :param painter: painter object
        :param frame: Image object
        :return: None
        """
        self.label = Label(text="Predicted number: " + str(predicted_number) + ", prediction accuracy: " +
                                str(accuracy) + "\n" + "Was the prediction correct?",
                           size_hint=(1, None), font_size=15)
        self.decision_holder = BoxLayout(orientation="horizontal", size_hint=(1, 1))
        button = Button(text="Good")
        button.bind(on_release=partial(self.good_prediction, painter, frame))
        self.decision_holder.add_widget(button)
        button = Button(text="Bad")
        button.bind(on_release=partial(self.bad_prediction, painter, frame))
        self.decision_holder.add_widget(button)
        self.box_layout.add_widget(self.label)
        self.box_layout.add_widget(self.decision_holder)

    def _clear_widgets(self):
        """
        Clear widgets from objects
        :return: None
        """
        try:
            self.box_layout.remove_widget(self.label)
            self.box_layout.remove_widget(self.decision_holder)
        except Exception as e:
            pass

    def good_prediction(self, painter, frame, obj):
        """
        Happens if prediction was good
        :param painter: painter object
        :param frame: Image object
        :param obj: button object
        :return: None
        """
        painter.canvas.clear()
        frame.alpha = 0
        frame.source = ""
        frame.reload()
        self._clear_widgets()

    def bad_prediction(self, painter, frame, obj):
        """
        Happens if prediction was bad
        :param painter: painter object
        :param frame: Image object
        :param obj: button object
        :return: None
        """
        self.label.text = "Insert a digit that you have painted"
        self.decision_holder.clear_widgets()
        self.text_input = TextInput(hint_text="Insert HERE a digit that you have painted and click Submit",
                                    size_hint=(1, None))
        self.decision_holder.add_widget(self.text_input)
        button = Button(text="Submit")
        self.decision_holder.add_widget(button)
        button.bind(on_release=partial(self.save_bad_predicted_image, painter, frame))

    def save_bad_predicted_image(self, painter, frame, obj):
        """
        Save image if prediction was bad
        :param painter: painter object
        :param frame: Image object
        :param obj: button object
        :return:
        """
        file_name = str(self.text_input.text) + "." + str(time()).split(".")[0] + ".png"
        os.rename('../temporary_img.png', f"./SavedDigits/{self.text_input.text}/{file_name}.png")
        self._clear_widgets()
        painter.canvas.clear()
        frame.alpha = 0
        frame.source = ""
        frame.reload()


class SaveScreen(Screen):
    box_layout = ObjectProperty()

    def save_cropped_image(self, obj, text, painter, frame):
        """
        Save cropped image
        :param obj: button object
        :param text: text from TextInput object
        :param painter: painter object
        :param frame: Image object
        :return:
        """
        image = crop_image(painter)
        frame.alpha = 1
        frame.source = "temporary_img.png"
        frame.reload()
        file_name = str(text.text) + "." + str(time()).split(".")[0] + ".png"
        cv.imwrite(f"./SavedDigits/{text.text}/{file_name}", image)
        painter.canvas.clear()


class PaintWidget(Widget):

    def on_touch_down(self, touch):
        """
        Happens when mouse is clicked
        :param touch: touch object
        :return: None
        """
        if self.collide_point(*touch.pos):
            with self.canvas:
                Color((1, 1, 1))
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=6.8)

    def on_touch_move(self, touch):
        """
        Happens when clicked mouse if moving
        :param touch: touch object
        :return: None
        """
        if self.collide_point(*touch.pos):
            touch.ud['line'].points += [touch.x, touch.y]


class App(App):
    width = NumericProperty()
    height = NumericProperty()

    def build(self):
        """
        Build the App
        :return: application
        """
        self.width = Window.width
        self.height = Window.height
        global MODEL
        MODEL = load_model(MODEL_NAME)
        app = ScreenManager(transition=NoTransition())
        app.add_widget(MainScreen(name='Main'))
        app.add_widget(SaveScreen(name='Train'))
        return app

    def export_widget_to_file(self, textinput, widget):
        """
        Export screenshot of widget to file
        :param textinput: TextInput object
        :param widget: Painter object
        :return: None
        """
        if textinput.text != "":
            file_name = str(textinput.text) + "." + str(time()).split(".")[0] + ".png"
            widget.export_to_png(f"./SavedDigits/{textinput}/{file_name}")
            widget.canvas.clear()

    def clear_canvas(self, obj, frame):
        """
        Clear painting area
        :param obj: Object
        :param frame: Painter object
        :return: None
        """
        obj.canvas.clear()
        frame.source = ""
        frame.alpha = 0
        frame.reload()


if __name__ == '__main__':
    App().run()
