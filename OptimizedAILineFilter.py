import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class FiltroIA:
    def __init__(self, root): 
        self.root = root
        self.root.title("Filtro de Linhas com IA")
        self.root.geometry("600x500")

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=70, height=15)
        self.text_area.pack(pady=10)

        self.keyword_entry = tk.Entry(self.root, font=("Arial", 12), width=30)
        self.keyword_entry.pack(pady=5)

        self.load_button = tk.Button(self.root, text="Carregar Arquivo", command=self.load_file)
        self.load_button.pack(pady=5)

        self.filter_button = tk.Button(self.root, text="Filtrar Linhas", command=self.filter_lines)
        self.filter_button.pack(pady=5)

        self.ai_button = tk.Button(self.root, text="Otimizar com IA", command=self.optimize_with_ai)
        self.ai_button.pack(pady=5)

        self.save_button = tk.Button(self.root, text="Salvar Resultado", command=self.save_file)
        self.save_button.pack(pady=5)

        self.lines = []

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Arquivos de Texto", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.lines = file.readlines()
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, "".join(self.lines))

    def filter_lines(self):
        keyword = self.keyword_entry.get().strip().lower()
        if not keyword:
            messagebox.showwarning("Aviso", "Digite uma palavra-chave para filtrar.")
            return
        
        filtered_lines = [line for line in self.lines if keyword in line.lower()]
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "".join(filtered_lines))

    def optimize_with_ai(self):
        if not self.lines:
            messagebox.showwarning("Aviso", "Carregue um arquivo antes de usar a IA.")
            return
        
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(self.lines)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        optimized_lines = [self.lines[i] for i in range(len(self.lines)) if clusters[i] == 1]
        
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "".join(optimized_lines))

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Arquivos de Texto", "*.txt")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(self.text_area.get("1.0", tk.END))
            messagebox.showinfo("Sucesso", "Arquivo salvo com sucesso!")

if __name__ == "__main__": 
    root = tk.Tk()
    app = FiltroIA(root)
    root.mainloop()
