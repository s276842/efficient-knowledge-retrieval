class Encoder():
    def __init__(self, model, tokenizer, preprocessing_transform=None, postprocessing_transform=None):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessing_transform = preprocessing_transform
        self.postprocessing_transform = postprocessing_transform

    def __call__(self, item):

        try:
            iter(item)
        except:
            item = [item]

        if self.preprocessing_transform is not None:
            x = [self.preprocessing_transform(val) for val in item]

        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        x = self.model(**x)

        if self.postprocessing_transform is not None:
            x = self.postprocessing_transform(x)

        return x
