import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]

if __name__ == '__main__':
    if arg1 == 'an':
        # get raw corpus
        # list of UBMS terms (unambigious)
        # function to match words (i.e. annotating those that don't have multiple semantic types)
        # model trained binary classifier to distinguish postive or negative examples
    if arg1 == 'train':
        # data preprocessor - pull out all annnotations and lemmatize them
        # create model decision forest classifier 
        # refer back to annotated corpus to create more annotations
    if arg1 == 'bd':
        # feature assembly - pos/neg examples, positions, lemmas, stems, POS tags, UMLS types
        # learning code to discover what features can extend boundaries (same decision forest)
        # run classification 'scorer' that determines threshold
    if arg1 == 'ev' and arg2 == 'strict':
        # load i2b2 2010 Dataset
        # calc function of precision (strict, only counts perfect matches)
        # calc function of recall (strict)
        # calc function of F1 (strict)
    if arg1 == 'ev' and arg2 == 'relaxed':
        # load i2b2 2010 Dataset
        # calc function of precision (relaxed, matches) 
        # calc function of recall (relaxed)
        # calc function of F1 (relaxed)
    

