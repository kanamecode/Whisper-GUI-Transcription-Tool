import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import whisper
from pydub import AudioSegment, silence
import os
import torch
import threading
import queue

# Language map
language_map = {
    "Japanese": "ja",
    "English": "en",
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
}

def async_typewriter(widget, text_queue, full_text_buffer, done_event):
    current_text = None
    current_index = 0
    stop_typing = False
    loading_text_base = "Processing"
    loading_dots = 0
    loading_line_index = None
    loading_cleared = False

    def process_queue():
        nonlocal current_text, current_index, stop_typing
        nonlocal loading_dots, loading_line_index, loading_cleared

        if stop_typing:
            return

        if current_text is None and text_queue.empty() and not done_event.is_set():
            if loading_line_index is None:
                widget.insert(tk.END, loading_text_base + "...\n")
                loading_line_index = float(widget.index("end-2l").split('.')[0])
            else:
                loading_dots = (loading_dots + 1) % 6
                dots_str = "." * loading_dots
                display_text = loading_text_base + dots_str + " " * (3 - loading_dots)
                widget.delete(f"{int(loading_line_index)}.0", f"{int(loading_line_index)}.end")
                widget.insert(f"{int(loading_line_index)}.0", display_text)
            widget.see(tk.END)
            widget.after(500, process_queue)
            return

        if not loading_cleared and not text_queue.empty():
            if loading_line_index is not None:
                widget.delete(f"{int(loading_line_index)}.0", f"{int(loading_line_index)}.end+1c")
                loading_line_index = None
            loading_cleared = True

        if done_event.is_set():
            stop_typing = True
            if not loading_cleared and loading_line_index is not None:
                widget.delete(f"{int(loading_line_index)}.0", f"{int(loading_line_index)}.end+1c")
            full_text = "\n".join(full_text_buffer)
            widget.delete(1.0, tk.END)
            widget.insert(tk.END, full_text)
            widget.see(tk.END)
            messagebox.showinfo("Done", "Transcription completed.")
            start_button.config(
                state=tk.NORMAL,
                text="Start Transcription",
                bg="green",
                fg="white"
            )
            return

        if current_text is None:
            if not text_queue.empty():
                current_text = text_queue.get()
                current_index = 0
            else:
                widget.after(100, process_queue)
                return

        if current_text is not None:
            if current_index < len(current_text):
                widget.insert(tk.END, current_text[current_index])
                widget.see(tk.END)
                current_index += 1
                widget.after(30, process_queue)
            else:
                widget.insert(tk.END, "\n")
                widget.see(tk.END)
                current_text = None
                current_index = 0
                widget.after(100, process_queue)

    process_queue()

def transcribe_realtime(file_path, language_code, device_choice, model_choice, output_folder, text_queue, full_text_buffer, done_event):
    try:
        if device_choice == "GPU" and not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is not available.")

        device = "cuda" if device_choice == "GPU" else "cpu"
        model = whisper.load_model(model_choice, device=device)

        audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(16000)

        chunks = silence.split_on_silence(
            audio,
            min_silence_len=1000,
            silence_thresh=audio.dBFS - 16,
            keep_silence=300
        )

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, base_name + ".txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                chunk_path = f"temp_chunk_{i}.wav"
                chunk.export(chunk_path, format="wav")
                result = model.transcribe(chunk_path, language=language_code)
                os.remove(chunk_path)
                text = result["text"].strip()
                f.write(text)
                full_text_buffer.append(text)
                text_queue.put(text)

        done_event.set()

    except Exception as e:
        messagebox.showerror("Error", str(e))
        start_button.config(state=tk.NORMAL)
    finally:
        if device_choice == "GPU":
            torch.cuda.empty_cache()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        if not output_entry.get():
            default_folder = os.path.dirname(file_path)
            output_entry.delete(0, tk.END)
            output_entry.insert(0, default_folder)

def browse_output():
    folder = filedialog.askdirectory()
    if folder:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, folder)

def start_transcription():
    file_path = file_entry.get()
    folder = output_entry.get()
    language = language_map.get(lang_var.get(), "ja")
    device = device_var.get()
    model = model_var.get()

    if not file_path:
        messagebox.showwarning("Warning", "Please select an audio file.")
        return

    if not folder:
        folder = os.path.dirname(file_path)
        output_entry.delete(0, tk.END)
        output_entry.insert(0, folder)

    text_output.delete(1.0, tk.END)
    start_button.config(state=tk.DISABLED, text="Transcribing...", bg="#FFA500", fg="white")
    root.update_idletasks()

    text_queue = queue.Queue()
    full_text_buffer = []
    done_event = threading.Event()

    async_typewriter(text_output, text_queue, full_text_buffer, done_event)

    thread = threading.Thread(
        target=transcribe_realtime,
        args=(file_path, language, device, model, folder, text_queue, full_text_buffer, done_event)
    )
    thread.start()

# ================== UI ==================
root = tk.Tk()
root.title("Audio Transcription Tool")
root.geometry("800x600")

for i in range(10):
    root.rowconfigure(i, weight=1)
for i in range(3):
    root.columnconfigure(i, weight=1)

tk.Label(root, text="Audio File:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
file_entry = tk.Entry(root)
file_entry.grid(row=0, column=1, sticky="ew", padx=5)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, sticky="w", padx=5)

tk.Label(root, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
output_entry = tk.Entry(root)
output_entry.grid(row=1, column=1, sticky="ew", padx=5)
tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2, sticky="w", padx=5)

tk.Label(root, text="Output Language:").grid(row=2, column=0, sticky="e", padx=5)
lang_var = tk.StringVar(value="Japanese")
lang_menu = ttk.Combobox(root, textvariable=lang_var, values=list(language_map.keys()), state="readonly")
lang_menu.grid(row=2, column=1, sticky="ew", padx=5)

tk.Label(root, text="Device:").grid(row=3, column=0, sticky="e", padx=5)
device_var = tk.StringVar(value="CPU")
device_menu = ttk.Combobox(root, textvariable=device_var, values=["CPU", "GPU"], state="readonly")
device_menu.grid(row=3, column=1, sticky="ew", padx=5)

tk.Label(root, text="Model:").grid(row=4, column=0, sticky="e", padx=5)
model_var = tk.StringVar(value="base")
model_menu = ttk.Combobox(root, textvariable=model_var,
                          values=["tiny", "base", "small", "medium", "large"], state="readonly")
model_menu.grid(row=4, column=1, sticky="ew", padx=5)

start_button = tk.Button(root, text="Start Transcription", bg="green", fg="white", command=start_transcription, font=("Helvetica", 17))
start_button.grid(row=5, column=1, sticky="ew", padx=100, pady=10)

text_output = tk.Text(root, wrap="word", height=20)
text_output.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

root.mainloop()
