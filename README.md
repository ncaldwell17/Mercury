# Mercury

Mercury is a Python-based application that runs on a sequence-based named entity recognition machine learning model and work pipeline designed to locate, extract & preprocess, classify, analyze, and pull useful information from clinical white papers. 

## Features
`locator.py` = an RSS feed-like Python module that allows the user to assemble a list of candidate clinical white papers from self-curated or built-in medical resources and save them to a local or virtual machine as PDFs while outputting information about the PDFs. The input of the `locator()` class is a __list__ of candidate sites, and the two outputs are __a save operation__ and the creation of a __dictionary__ of __dictionaries__. The first dictionary contains an inclusive index (an integer from 1-n documents) as its key and the associated PDFs as its values - each of the __associated PDFs__ is also a dictionary, containing the keys: title, path, and site-of-origin and their eponymous qualities as the values. _Future work involves the inclusion of additional keys in the inner dictionaries to indicate the subject area, length, etc._
  * NOTES: This module does not have any dependencies, and can run on its own with the listed packages installed within the Python interpreter. 
  * Required Packages:
    - pandas
    
`extractor.py` = a list of functions to take a .pdf file and extract the text from it and save the text as .txt files. _Future work includes the full integration of these functions with a main pipeline that can handle the automatic extraction of pulled PDFs_
  * Required Packages:
    - os
    - pdfminer
    - io

`.classifier.py` = a binary classifier machine learning module that allows the user to take the output of the `locator()` class and classify each of the PDFs as valid or invalid research. This is meant to curate our PDF collection to only include papers that assist in our objective. The  `classifier()` class is meant to be as flexible as possible to allow additional users to customize their own free parameters. The input of the classifier is a .txt file and the output is a boolean Y/N dictionary value assigned to a key in the previously created dictionary and the .txt content as another value for key content. All dictionary entries who are assigned an 'N' are dropped. 

`analyzer.py` = implements a version of fine-grained named-entity recognition, a method of NER that was first introduced in 2012 [1] that allows for a wider range of tasks such as multi-class, multi-label classification and unsupervised collection of training data. QS plans on integrating current state-of-the-art improvements to the above model, FIGER, [1] with SciBERT [2], an extension of a breakthrough 2018 research project BERT [3] that investigated the use of transfer-learning through pre-trained word embeddings to improve the precision of downstream NLP tasks, such as NER. This technology is made possible through developments in the TensorFlow API developed by Google that enables high-speed processing of the multi-dimensional matrix multiplication that is necessary to identify abstract relationships between words, sentences, and documents that cannot be identified by human readers. QS will focus on initially developing a successor to FIGER that focuses on our exact use case and amplified by SciBERT to conduct the first phase of our extraction engine, to be later improved by developing domain-specific corpora in the spirit of SciBERT and unsupervised relation tagging developed by our improvements to the FIGER model.

`pull.py` = transmits and organizes the outputs of `analyzer.py` to a graph database. 

# Use
This section is still under development. 


## Sources
[1] Xiao Ling and Daniel S. Weld. 2012. Fine-grained entity recognition. In Proceedings of the Twenty-Sixth AAAI Conference on Artificial Intelligence(AAAI'12). AAAI Press 94-100.
[2] Iz Beltagy, Arman Cohan, and Kyle Lo. 2019. Scibert: Pretrained contextualized embeddings for scientific text. CoRR, abs/1903.10676.
[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805.
