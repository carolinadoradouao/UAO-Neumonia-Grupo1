"""Interfaz gráfica de la herramienta de detección rápida de neumonía.

La vista mantiene el comportamiento original basado en Tkinter y delega toda
la lógica de negocio en ``src.integrator``. Este módulo no importa TensorFlow,
Keras ni ``load_model``.
"""

from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

from PIL import Image, ImageGrab, ImageTk

from src import integrator


class App:
    """Ventana principal para cargar imágenes, predecir y generar reportes."""

    def __init__(self, root=None):
        """Construye la ventana y todos los controles de la interfaz.

        Args:
            root: Ventana Tkinter opcional. Si no se proporciona, se crea una
                nueva instancia de ``tk.Tk``.
        """
        self.root = root if root is not None else tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        fonti = font.Font(root=self.root, weight="bold")

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = tk.StringVar(master=self.root)
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.text_img1 = tk.Text(self.root, width=31, height=15)
        self.text_img2 = tk.Text(self.root, width=31, height=15)
        self.text2 = tk.Text(self.root)
        self.text3 = tk.Text(self.root)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array = None
        self.img1 = None
        self.img2 = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.report_id = 0

    def _photo_image(self, image, size=(250, 250)):
        """Convierte una imagen PIL o un arreglo en una imagen para Tkinter.

        Args:
            image: Imagen PIL o arreglo de NumPy que se desea mostrar.
            size: Tamaño final ``(ancho, alto)`` de la imagen.

        Returns:
            Una instancia de ``ImageTk.PhotoImage`` lista para insertarse en un
            widget de texto.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        resized = image.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def load_img_file(self):
        """Abre un diálogo, carga la imagen seleccionada y la muestra."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if not filepath:
            return

        self.array, img2show = integrator.load_image(filepath)
        self.img1 = self._photo_image(img2show)
        self.text_img1.delete("1.0", tk.END)
        self.text_img1.image_create(tk.END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self):
        """Ejecuta la predicción y muestra la clase, probabilidad y heatmap."""
        if self.array is None:
            showinfo(title="Predecir", message="Primero debe cargar una imagen.")
            return

        self.label, self.proba, self.heatmap = integrator.predict(self.array)
        self.img2 = self._photo_image(self.heatmap)
        self.text_img2.delete("1.0", tk.END)
        self.text_img2.image_create(tk.END, image=self.img2)
        self.text2.delete("1.0", tk.END)
        self.text2.insert(tk.END, self.label)
        self.text3.delete("1.0", tk.END)
        self.text3.insert(tk.END, f"{self.proba:.2f}%")

    def save_results_csv(self):
        """Guarda la cédula y el resultado actual en ``historial.csv``."""
        if self.label is None:
            showinfo(title="Guardar", message="Primero debe realizar una predicción.")
            return

        with open("historial.csv", "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([self.text1.get(), self.label, f"{self.proba:.2f}%"])
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        """Captura la ventana actual y guarda el reporte como PDF."""
        self.root.update()
        left = self.root.winfo_rootx()
        top = self.root.winfo_rooty()
        right = left + self.root.winfo_width()
        bottom = top + self.root.winfo_height()
        capture = ImageGrab.grab(bbox=(left, top, right, bottom))

        pdf_path = f"Reporte{self.report_id}.pdf"
        capture.convert("RGB").save(pdf_path, "PDF", resolution=100.0)
        self.report_id += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Limpia los campos y las imágenes después de confirmar la acción."""
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if not answer:
            return

        self.text1.delete(0, tk.END)
        self.text2.delete("1.0", tk.END)
        self.text3.delete("1.0", tk.END)
        self.text_img1.delete("1.0", tk.END)
        self.text_img2.delete("1.0", tk.END)
        self.array = None
        self.img1 = None
        self.img2 = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.button1["state"] = "disabled"
        showinfo(title="Borrar", message="Los datos se borraron con éxito")

    def run(self):
        """Inicia el bucle principal de la interfaz gráfica."""
        self.root.mainloop()


def main():
    """Punto de entrada de la interfaz gráfica."""
    App().run()
    return 0


if __name__ == "__main__":
    main()
