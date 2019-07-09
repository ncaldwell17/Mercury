from flask import Flask, render_template
# Extractor module
import extractor
# Visualizer module
import visualizer


def main():
    text = extractor.extract_pdf_text_from_file()
    visualizer.run_example(str(text))


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('example2.html')


if __name__ == "__main__":
    main()
    app.run(debug=True)
