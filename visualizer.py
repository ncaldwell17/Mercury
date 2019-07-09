import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
from pathlib import Path
from flask import Flask, render_template
nlp = spacy.load('en')


def visualize_ner(a_string):
    # all colors can be associated with a tag as a key/value dict
    #   the format = {"TAG" : "color"}, which overrides the TAG
    colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    # all options can be associated with a parameter as a k/v dict
    #   format = {"ents": ["TAG"], "colors": colors}
    options = {"ents": ["ORG"], "colors": colors}
    visualization = displacy.render(nlp(a_string),
                                    jupyter=None,
                                    # indicates that I want to visualize entities
                                    style="ent",
                                    # I can use the options to alter the look
                                    # options=options,
                                    # wraps the output up as an html file for display
                                    page=True)
    return visualization


def run_example():
    ex_string = 'The p56Lck inhibitor Dasatinib was shown to enhance apoptosis induction by dexamethasone in otherwise GC-resistant CLL cells. This finding concurs with the observation by Sade showing that Notch-mediated resistance of a mouse lymphoma cell line could be overcome by inhibiting p56Lck.'
    html = visualize_ner(ex_string)
    output_path = Path("templates/example4.html")
    output_path.open("w", encoding="utf-8").write(html)

# uncomment this and delete the positional argument to run by itself
run_example()


#app = Flask(__name__)


# @app.route("/")
#def display():
    # return render_template('example.html')


#if __name__ == "__main__":
    #app.run(debug=True)
