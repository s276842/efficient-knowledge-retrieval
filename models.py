class Encoder():
    def __init__(self, model, tokenizer, preprocessing_transform=None, postprocessing_transform=None):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessing_transform = preprocessing_transform
        self.postprocessing_transform = postprocessing_transform

    #todo implement batches
    def __call__(self, item):

        try:
            iter(item)
            data = item
        except:
            data = [item]

        if self.preprocessing_transform is not None:
        #     x = [self.preprocessing_transform(val) for val in item]
            data = self.preprocessing_transform(data)


        tokenized_data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        tokenized_data = {key:value.to(self.model.device) for key, value in tokenized_data.items()}
        out = self.model(**tokenized_data)
        del tokenized_data, data

        if self.postprocessing_transform is not None:
            out = self.postprocessing_transform(out)

        return out
