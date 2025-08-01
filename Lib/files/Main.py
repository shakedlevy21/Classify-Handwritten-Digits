import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tkinter import *
from PIL import ImageGrab, Image

#disable TF2 optimizations that spam warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#load MNIST using Keras (replaces deprecated input_data)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#normalize and flatten images for classic ML
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

print("Training data:", x_train.shape, y_train.shape)
print("Test data:", x_test.shape, y_test.shape)

# show a digit
# plt.imshow(x_train[1].reshape(28, 28), cmap='gray')
# plt.title(f"Label: {y_train[1]}")
# plt.show()

#classification
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining model...")
model.fit(x_train, y_train, epochs=4, batch_size=32, verbose=1)

#evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {acc*100:.2f}%")

#drawing app function
def draw_digit():
    window = Tk()
    window.title("Draw a digit (0-9)")

    canvas = Canvas(window, width=200, height=200, bg='black')
    canvas.pack()

    #draw white line
    def paint(event):
        x1, y1 = (event.x-6), (event.y-6)
        x2, y2 = (event.x+6), (event.y+6)
        canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')

    canvas.bind("<B1-Motion>", paint)

    def predict():
        #capture canvas content
        x = window.winfo_rootx() + canvas.winfo_x()
        y = window.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()

        #grab image and preprocess
        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.resize((28, 28)).convert('L')
        img = np.array(img)
        img = 255 - img  # Invert colors: white digit on black
        img = img.reshape(1, 784).astype('float32') / 255.0

        #predict digit
        prediction = np.argmax(model.predict(img))
        print(f"Model predicts: {prediction}")

    btn_predict = Button(window, text="Predict", command=predict)
    btn_predict.pack()

    btn_clear = Button(window, text="Clear", command=lambda: canvas.delete("all"))
    btn_clear.pack()

    window.mainloop()

#main
while True:
    choice = input("\nType 'test' to test dataset images, 'draw' to draw a digit, or 'q' to quit: ").lower()

    if choice == 'q':
        break
    elif choice == 'draw':
        draw_digit()
    elif choice == 'test':
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx].reshape(1, 784)
        prediction = np.argmax(model.predict(img))
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {prediction}, True: {y_test[idx]}")
        plt.show()
    else:
        print("Invalid input!")