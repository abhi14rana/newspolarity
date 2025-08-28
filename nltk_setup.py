import nltk
import os
import subprocess

NLTK_DIR = "/opt/render/nltk_data"

def download_all():
    os.makedirs(NLTK_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DIR)

    # Run textblob’s downloader (this is a script, not a function)
    subprocess.run(["python", "-m", "textblob.download_corpora"], check=True)

    # Explicit NLTK downloads
    for pkg in [
        "punkt",
        "punkt_tab",  # 👈 required for TextBlob/NRClex
        "averaged_perceptron_tagger",
        "brown",
        "wordnet",
        "omw-1.4"
    ]:
        nltk.download(pkg, download_dir=NLTK_DIR)

if __name__ == "__main__":
    download_all()
