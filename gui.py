import tkinter as tk
from Generator import Generator

NUMBER_OF_SUGGESTIONS = 3
TRANSFORMER_MODEL_FILE = "" ##INSERT HERE the trained transformer file

NGRAM_MODEL_FILES = {
    ("openwebtext", "bigram"):  "openwebtext_bigram.txt",
    ("openwebtext", "trigram"): "openwebtext_trigram.txt",
    ("wikitext103",  "bigram"):  "wiki103_model_ngram.txt",
    ("wikitext103",  "trigram"): "wiki103_model_trigram.txt",
    ("wikitext2",   "bigram"):  "Wikitext2_model_ngram.txt",
    ("wikitext2",   "trigram"): "Wikitext2_model_trigram.txt",
}

class NgramPredictorWrapper:
    def __init__(self, model_file):
        self.model = Generator()
        self.model.read_model(model_file)

    def predict(self, previous_word, prefix, k):
        if previous_word is None:
            return self.model.predict_by_unigram(prefix, k)
        else:
            return self.model.predict_next_words(previous_word, prefix, k)


class TransformerPredictorWrapper:
    def __init__(self, model_file=None):
        print("Transformer model selected")

    def predict(self, previous_word, prefix, k): ###Tong this one is not filled yet, just fill in later
        dummy_words = ["transformer", "prediction", "model"]

        suggestions = []

        for word in dummy_words:
            if prefix == "" or word.startswith(prefix.lower()):
                suggestions.append(word)

        return suggestions[:k]


class ModelSelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Choose Prediction Model")

        tk.Label(root, text="Choose which model:", font=("Cambria", 16)).pack(pady=20)

        tk.Button(
            root,
            text="Use N-gram Model",
            font=("Cambria", 14),
            width=25,
            command=self.open_dataset_selection
        ).pack(pady=10)

        tk.Button(
            root,
            text="Use Transformer Model",
            font=("Cambria", 14),
            width=25,
            command=self.open_transformer_gui
        ).pack(pady=10)

    def open_dataset_selection(self):
        self.root.destroy()
        new_root = tk.Tk()
        DatasetSelectionGUI(new_root)
        new_root.mainloop()

    def open_transformer_gui(self):
        predictor = TransformerPredictorWrapper(TRANSFORMER_MODEL_FILE)
        self.root.destroy()

        new_root = tk.Tk()
        WordPredictionGUI(new_root, predictor, "Word Prediction Using Transformer Model")
        new_root.mainloop()


class DatasetSelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Choose Dataset")

        tk.Label(root, text="Choose training dataset:", font=("Cambria", 16)).pack(pady=20)

        datasets = [
            ("OpenWebText", "openwebtext"),
            ("Wikitext-103", "wikitext103"),
            ("Wikitext-2",   "wikitext2"),
        ]

        for label, key in datasets:
            tk.Button(
                root,
                text=label,
                font=("Cambria", 14),
                width=25,
                command=lambda k=key: self.open_gram_selection(k)
            ).pack(pady=10)

    def open_gram_selection(self, dataset_key):
        self.root.destroy()
        new_root = tk.Tk()
        GramSelectionGUI(new_root, dataset_key)
        new_root.mainloop()


class GramSelectionGUI:
    def __init__(self, root, dataset_key):
        self.root = root
        self.dataset_key = dataset_key
        self.root.title("Choose N-gram Order")

        dataset_labels = {
            "openwebtext": "OpenWebText",
            "wikitext103":  "Wikitext-103",
            "wikitext2":   "Wikitext-2",
        }
        label_text = f"Dataset: {dataset_labels[dataset_key]}\nChoose n-gram order:"
        tk.Label(root, text=label_text, font=("Cambria", 16), justify="center").pack(pady=20)

        tk.Button(
            root,
            text="Bigram",
            font=("Cambria", 14),
            width=25,
            command=lambda: self.launch("bigram")
        ).pack(pady=10)

        tk.Button(
            root,
            text="Trigram",
            font=("Cambria", 14),
            width=25,
            command=lambda: self.launch("trigram")
        ).pack(pady=10)

    def launch(self, gram_key):
        model_file = NGRAM_MODEL_FILES[(self.dataset_key, gram_key)]
        predictor = NgramPredictorWrapper(model_file)

        dataset_labels = {
            "openwebtext": "OpenWebText",
            "wikitext103":  "Wikitext-103",
            "wikitext2":   "Wikitext-2",
        }
        title = f"N-gram ({gram_key.capitalize()}) — {dataset_labels[self.dataset_key]}"

        self.root.destroy()
        new_root = tk.Tk()
        WordPredictionGUI(new_root, predictor, title)
        new_root.mainloop()


class WordPredictionGUI:
    def __init__(self, root, predictor, title_text):
        self.root = root
        self.root.title("Word Predictor")

        self.predictor = predictor

        self.label = tk.Label(
            root,
            text=title_text,
            font=("Cambria", 14)
        )
        self.label.pack(pady=5)

        self.text_box = tk.Text(
            root,
            height=4,
            width=60,
            font=("Cambria", 14)
        )
        self.text_box.pack(padx=10, pady=10)

        self.text_box.bind("<KeyRelease>", self.on_text_changed)
        self.text_box.bind("<Tab>", self.autofill_best_suggestion)

        self.suggestion_frame = tk.Frame(root)
        self.suggestion_frame.pack(pady=10)

        self.suggestion_buttons = []
        self.display_order = [1, 0, 2]

        for button_position in range(NUMBER_OF_SUGGESTIONS):
            suggestion_index = self.display_order[button_position]

            button = tk.Button(
                self.suggestion_frame,
                text="",
                font=("Cambria", 12),
                width=12,
                command=lambda index=suggestion_index: self.insert_suggestion(index)
            )

            button.grid(row=0, column=button_position, padx=5)
            self.suggestion_buttons.append(button)

        self.current_suggestions = []

    def get_context_and_prefix(self):
        text = self.text_box.get("1.0", "end-1c")

        if not text.strip():
            return None, ""

        words = text.split()

        if text.endswith(" "):
            previous_word = words[-1]
            prefix = ""
            return previous_word, prefix

        prefix = words[-1]

        if len(words) >= 2:
            previous_word = words[-2]
        else:
            previous_word = None

        return previous_word, prefix

    def on_text_changed(self, event=None):
        previous_word, prefix = self.get_context_and_prefix()

        suggestions = self.predictor.predict(
            previous_word,
            prefix,
            NUMBER_OF_SUGGESTIONS
        )

        self.current_suggestions = suggestions
        self.update_buttons(suggestions)

    def update_buttons(self, suggestions):
        for button_position, button in enumerate(self.suggestion_buttons):
            suggestion_index = self.display_order[button_position]

            if suggestion_index < len(suggestions):
                button.config(text=suggestions[suggestion_index], state="normal")
            else:
                button.config(text="", state="disabled")

    def insert_suggestion(self, index):
        if index >= len(self.current_suggestions):
            return

        suggestion = self.current_suggestions[index]

        text = self.text_box.get("1.0", "end-1c")

        if not text:
            self.text_box.insert("end", suggestion + " ")
            self.on_text_changed()
            return

        if text.endswith(" "):
            self.text_box.insert("end", suggestion + " ")
            self.on_text_changed()
            return

        words = text.split()
        prefix = words[-1]

        chars_to_delete = len(prefix)
        self.text_box.delete(f"end-{chars_to_delete + 1}c", "end-1c")
        self.text_box.insert("end", suggestion + " ")

        self.on_text_changed()

    def autofill_best_suggestion(self, event=None):
        if not self.current_suggestions:
            return "break"

        self.insert_suggestion(0)

        return "break"


def main():
    root = tk.Tk()
    ModelSelectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()