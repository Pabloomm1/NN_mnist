import numpy as np
import re
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
from datetime import datetime


def load_parameter(filename, expected_shape=None):
    with open(filename, 'r') as f:
        content = f.read()
    content = re.sub(r'[\[\]]', '', content)
    numbers = [float(x) for x in content.split()]
    arr = np.array(numbers)
    if expected_shape:
        arr = arr.reshape(expected_shape)
    return arr


def load_weights_from_txt():
    w1_flat = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/W1.txt")
    W1 = w1_flat.reshape(128, 784)

    w2_flat = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/W2.txt")
    W2 = w2_flat.reshape(64, 128)

    w3_flat = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/W3.txt")
    W3 = w3_flat.reshape(10, 64)

    b1 = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/b1.txt")
    b1 = b1.reshape(128, 1)

    b2 = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/b2.txt")
    b2 = b2.reshape(64, 1)

    b3 = load_parameter("/home/pavel/work/notebook/Mnist3/parameters/b3.txt")
    b3 = b3.reshape(10, 1)

    return W1, b1, W2, b2, W3, b3


def Relu(x):
    return np.maximum(0, x)


def SoftMax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1 @ X + b1
    A1 = Relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = Relu(Z2)
    Z3 = W3 @ A2 + b3
    A3 = SoftMax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def predict(W1, b1, W2, b2, W3, b3, X):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return np.argmax(A3, axis=0)


def canvas_to_mnist(pil_image):
    img = pil_image.convert('L')
    
   
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
   
    w, h = img.size
    pad = int(max(w, h) * 0.3)
    img = ImageOps.expand(img, border=pad, fill=0)
    
   
    img = img.resize((28, 28), Image.LANCZOS)
    
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.reshape(784, 1)


class DrawApp:
    CANVAS_SIZE = 480
    BRUSH_SIZE = 12

    def __init__(self, root, W1, b1, W2, b2, W3, b3):
        self.root = root
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

        self.root.title("MNIST Digit Recognizer")
        self.root.resizable(False, False)
        self.root.configure(bg='#1a1a1a')

        self._build_ui()
        self._reset_image()

    def _build_ui(self):
        pad = dict(padx=12, pady=8)

       
        tk.Label(self.root, text="Нарисуй цифру", font=('Helvetica', 14, 'bold'),
                 bg='#1a1a1a', fg='#ffffff').grid(row=0, column=0, columnspan=2, pady=(16, 4))

       
        self.tk_canvas = tk.Canvas(
            self.root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg='black', cursor='crosshair', highlightthickness=1,
            highlightbackground='#444444'
        )
        self.tk_canvas.grid(row=1, column=0, rowspan=6, padx=(16, 8), pady=8)
        self.tk_canvas.bind('<B1-Motion>', self._on_draw)
        self.tk_canvas.bind('<ButtonRelease-1>', self._on_release)
        self.last_x = None
        self.last_y = None

      
        tk.Label(self.root, text="Предсказание:", font=('Helvetica', 11),
                 bg='#1a1a1a', fg='#888888').grid(row=1, column=1, sticky='sw', **pad)

        self.digit_var = tk.StringVar(value='—')
        tk.Label(self.root, textvariable=self.digit_var, font=('Helvetica', 64, 'bold'),
                 bg='#1a1a1a', fg='#1db37e', width=3).grid(row=2, column=1, sticky='n', padx=12)

        self.conf_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self.conf_var, font=('Helvetica', 11),
                 bg='#1a1a1a', fg='#888888').grid(row=3, column=1, sticky='n', padx=12)

        
        tk.Label(self.root, text="Вероятности:", font=('Helvetica', 10),
                 bg='#1a1a1a', fg='#888888').grid(row=4, column=1, sticky='sw', padx=12, pady=(8, 2))

        self.bars_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.bars_frame.grid(row=5, column=1, sticky='nw', padx=12)
        self.bar_widgets = []
        for i in range(10):
            row_f = tk.Frame(self.bars_frame, bg='#1a1a1a')
            row_f.pack(fill='x', pady=1)
            lbl = tk.Label(row_f, text=str(i), width=2, font=('Helvetica', 9),
                           bg='#1a1a1a', fg='#666666', anchor='e')
            lbl.pack(side='left')
            track = tk.Canvas(row_f, width=120, height=8, bg='#2a2a2a',
                              highlightthickness=0)
            track.pack(side='left', padx=(4, 4))
            bar = track.create_rectangle(0, 0, 0, 8, fill='#333333', outline='')
            pct_lbl = tk.Label(row_f, text='', width=5, font=('Helvetica', 9),
                               bg='#1a1a1a', fg='#555555', anchor='w')
            pct_lbl.pack(side='left')
            self.bar_widgets.append((track, bar, pct_lbl))

       
        btn_frame = tk.Frame(self.root, bg='#1a1a1a')
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(4, 16))

        tk.Button(btn_frame, text='Очистить', command=self._clear,
                  font=('Helvetica', 11), bg='#2a2a2a', fg='#cccccc',
                  activebackground='#3a3a3a', activeforeground='#ffffff',
                  relief='flat', padx=16, pady=6, cursor='hand2').pack(side='left', padx=8)

        tk.Button(btn_frame, text='Предсказать', command=self._predict,
                  font=('Helvetica', 11, 'bold'), bg='#1d9e75', fg='#ffffff',
                  activebackground='#17b580', activeforeground='#ffffff',
                  relief='flat', padx=16, pady=6, cursor='hand2').pack(side='left', padx=8)

    def _reset_image(self):
        self.pil_image = Image.new('RGB', (self.CANVAS_SIZE, self.CANVAS_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.pil_image)

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = self.BRUSH_SIZE
        if self.last_x is not None:
            self.tk_canvas.create_line(self.last_x, self.last_y, x, y,
                                       fill='white', width=r * 2,
                                       capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill='white', width=r * 2)
        self.tk_canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='white')
        self.last_x, self.last_y = x, y

    def _on_release(self, event):
        self.last_x = None
        self.last_y = None

    def _clear(self):
        self.tk_canvas.delete('all')
        self._reset_image()
        self.digit_var.set('—')
        self.conf_var.set('')
        for track, bar, pct_lbl in self.bar_widgets:
            track.coords(bar, 0, 0, 0, 8)
            track.itemconfig(bar, fill='#333333')
            pct_lbl.config(text='')

    def _predict(self):
        X = canvas_to_mnist(self.pil_image)
        _, _, _, _, _, A3 = forward_prop(
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, X
        )
        probs = A3.flatten()
        best = int(np.argmax(probs))

        self.digit_var.set(str(best))
        self.conf_var.set(f'{probs[best] * 100:.1f}% уверенность')

        for i, (track, bar, pct_lbl) in enumerate(self.bar_widgets):
            w = int(probs[i] * 120)
            color = '#1db37e' if i == best else '#444444'
            track.coords(bar, 0, 0, w, 8)
            track.itemconfig(bar, fill=color)
            pct_lbl.config(text=f'{probs[i] * 100:.1f}%',
                           fg='#aaaaaa' if i == best else '#555555')


def main():
    try:
        W1, b1, W2, b2, W3, b3 = load_weights_from_txt()
    except Exception as e:
        messagebox.showerror("Ошибка загрузки весов", str(e))
        return

    root = tk.Tk()
    app = DrawApp(root, W1, b1, W2, b2, W3, b3)
    root.mainloop()


if __name__ == '__main__':
    main()
