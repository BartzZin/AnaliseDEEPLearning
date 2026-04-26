from __future__ import annotations

from pathlib import Path
from threading import Thread
from tkinter import BooleanVar, END, LEFT, RIGHT, BOTH, X, Y, Tk, Text, filedialog, messagebox, ttk

from mlp_experimento import executar_experimentos, resultados_para_csv


class AplicativoMLP(Tk):
    def __init__(self):
        super().__init__()
        self.title("Atividade 3 - MLP")
        self.geometry("980x680")
        self.minsize(860, 580)

        self.resultados = []
        self.relatorio = ""

        self.normalizado_var = BooleanVar(value=True)
        self.original_var = BooleanVar(value=True)

        self._configurar_estilo()
        self._criar_tela()

    def _configurar_estilo(self):
        estilo = ttk.Style(self)
        estilo.theme_use("clam")
        estilo.configure("TFrame", background="#f7f7f4")
        estilo.configure("TLabel", background="#f7f7f4", font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 16, "bold"))
        estilo.configure("TButton", font=("Segoe UI", 10))
        estilo.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _criar_tela(self):
        raiz = ttk.Frame(self, padding=16)
        raiz.pack(fill=BOTH, expand=True)

        topo = ttk.Frame(raiz)
        topo.pack(fill=X)

        ttk.Label(topo, text="Comparador simples de MLP By Lucas Lara", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(
            topo,
            text="Dataset Breast Cancer Wisconsin + MLPClassifier do scikit-learn",
        ).pack(anchor="w", pady=(4, 14))

        controles = ttk.Frame(raiz)
        controles.pack(fill=X, pady=(0, 12))

        ttk.Label(controles, text="Neuronios ocultos").grid(row=0, column=0, sticky="w")
        self.neuronios_entry = ttk.Entry(controles, width=24)
        self.neuronios_entry.insert(0, "5, 10, 20, 40")
        self.neuronios_entry.grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(4, 0))

        ttk.Label(controles, text="Taxas de aprendizagem").grid(row=0, column=1, sticky="w")
        self.taxas_entry = ttk.Entry(controles, width=24)
        self.taxas_entry.insert(0, "0.001, 0.01, 0.05")
        self.taxas_entry.grid(row=1, column=1, sticky="w", padx=(0, 16), pady=(4, 0))

        ttk.Checkbutton(controles, text="Testar dados normalizados", variable=self.normalizado_var).grid(
            row=1, column=2, sticky="w", padx=(0, 16)
        )
        ttk.Checkbutton(controles, text="Testar dados originais", variable=self.original_var).grid(
            row=1, column=3, sticky="w", padx=(0, 16)
        )

        botoes = ttk.Frame(raiz)
        botoes.pack(fill=X, pady=(0, 12))

        self.executar_btn = ttk.Button(botoes, text="Executar experimentos", command=self.executar)
        self.executar_btn.pack(side=LEFT)

        self.salvar_btn = ttk.Button(botoes, text="Salvar CSV", command=self.salvar_csv, state="disabled")
        self.salvar_btn.pack(side=LEFT, padx=(8, 0))

        self.status_label = ttk.Label(botoes, text="Pronto para executar.")
        self.status_label.pack(side=RIGHT)

        conteudo = ttk.PanedWindow(raiz, orient="vertical")
        conteudo.pack(fill=BOTH, expand=True)

        tabela_frame = ttk.Frame(conteudo)
        conteudo.add(tabela_frame, weight=2)

        colunas = (
            "neuronios",
            "taxa",
            "normalizado",
            "holdout",
            "cv",
            "tempo",
            "iteracoes",
        )
        self.tabela = ttk.Treeview(tabela_frame, columns=colunas, show="headings", height=9)
        cabecalhos = {
            "neuronios": "Neuronios",
            "taxa": "Taxa",
            "normalizado": "Normalizado",
            "holdout": "Holdout",
            "cv": "5-fold CV",
            "tempo": "Tempo (s)",
            "iteracoes": "Iteracoes",
        }
        for coluna, titulo in cabecalhos.items():
            self.tabela.heading(coluna, text=titulo)
            self.tabela.column(coluna, anchor="center", width=120)
        self.tabela.pack(side=LEFT, fill=BOTH, expand=True)

        barra_tabela = ttk.Scrollbar(tabela_frame, orient="vertical", command=self.tabela.yview)
        barra_tabela.pack(side=RIGHT, fill=Y)
        self.tabela.configure(yscrollcommand=barra_tabela.set)

        relatorio_frame = ttk.Frame(conteudo)
        conteudo.add(relatorio_frame, weight=3)

        self.relatorio_text = Text(relatorio_frame, wrap="word", font=("Consolas", 10), height=15)
        self.relatorio_text.pack(side=LEFT, fill=BOTH, expand=True)
        barra_relatorio = ttk.Scrollbar(relatorio_frame, orient="vertical", command=self.relatorio_text.yview)
        barra_relatorio.pack(side=RIGHT, fill=Y)
        self.relatorio_text.configure(yscrollcommand=barra_relatorio.set)

    def executar(self):
        try:
            neuronios = self._ler_inteiros(self.neuronios_entry.get())
            taxas = self._ler_floats(self.taxas_entry.get())
            if not self.normalizado_var.get() and not self.original_var.get():
                raise ValueError("Selecione pelo menos uma opcao de dados.")
        except ValueError as erro:
            messagebox.showerror("Entrada invalida", str(erro))
            return

        self.executar_btn.configure(state="disabled")
        self.salvar_btn.configure(state="disabled")
        self.status_label.configure(text="Executando MLP...")
        self._limpar_saida()

        Thread(
            target=self._executar_em_segundo_plano,
            args=(neuronios, taxas, self.normalizado_var.get(), self.original_var.get()),
            daemon=True,
        ).start()

    def _executar_em_segundo_plano(self, neuronios, taxas, usar_normalizado, usar_original):
        try:
            resultados, relatorio = executar_experimentos(
                neuronios=neuronios,
                taxas_aprendizado=taxas,
                usar_normalizado=usar_normalizado,
                usar_original=usar_original,
            )
            self.after(0, lambda: self._mostrar_resultados(resultados, relatorio))
        except Exception as erro:
            self.after(0, lambda: self._mostrar_erro(erro))

    def _mostrar_resultados(self, resultados, relatorio):
        self.resultados = resultados
        self.relatorio = relatorio

        for item in resultados:
            self.tabela.insert(
                "",
                END,
                values=(
                    item.neurons,
                    item.learning_rate,
                    "sim" if item.normalized else "nao",
                    f"{item.holdout_accuracy:.4f}",
                    f"{item.cv_mean_accuracy:.4f} +/- {item.cv_std_accuracy:.4f}",
                    f"{item.train_time:.3f}",
                    item.iterations,
                ),
            )

        self.relatorio_text.insert("1.0", relatorio)
        self.status_label.configure(text=f"{len(resultados)} experimentos finalizados.")
        self.executar_btn.configure(state="normal")
        self.salvar_btn.configure(state="normal")

    def _mostrar_erro(self, erro):
        self.status_label.configure(text="Erro ao executar.")
        self.executar_btn.configure(state="normal")
        messagebox.showerror("Erro", str(erro))

    def salvar_csv(self):
        if not self.resultados:
            return

        caminho = filedialog.asksaveasfilename(
            title="Salvar resultados",
            defaultextension=".csv",
            initialfile="resultados_mlp.csv",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return

        Path(caminho).write_text(resultados_para_csv(self.resultados), encoding="utf-8")
        self.status_label.configure(text=f"CSV salvo em {caminho}")

    def _limpar_saida(self):
        for item in self.tabela.get_children():
            self.tabela.delete(item)
        self.relatorio_text.delete("1.0", END)

    @staticmethod
    def _ler_inteiros(texto: str) -> list[int]:
        valores = [int(parte.strip()) for parte in texto.split(",") if parte.strip()]
        if not valores or any(valor <= 0 for valor in valores):
            raise ValueError("Informe neuronios positivos separados por virgula.")
        return valores

    @staticmethod
    def _ler_floats(texto: str) -> list[float]:
        if ";" in texto:
            partes = texto.split(";")
            valores = [float(parte.strip().replace(",", ".")) for parte in partes if parte.strip()]
        else:
            valores = [float(parte.strip()) for parte in texto.split(",") if parte.strip()]
        if not valores or any(valor <= 0 for valor in valores):
            raise ValueError("Informe taxas positivas. Exemplo: 0.001, 0.01, 0.05")
        return valores


if __name__ == "__main__":
    app = AplicativoMLP()
    app.mainloop()
