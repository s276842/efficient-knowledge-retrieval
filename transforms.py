
class CLSExtractor:
    def __call__(self, out):
        return out.last_hidden_state[:, 0]


class ConcatenateKnowledge:
    def __init__(self, sep_token=' ', add_domain=False, add_entity_name=False, add_answer=True, separate_question_answer=False):
        self.sep_token = ' ' + sep_token + ' ' if sep_token != ' ' else sep_token
        self.add_domain = add_domain
        self.add_entity_name = add_entity_name
        self.add_answer = add_answer
        self.separate_question_answer = separate_question_answer

    def __call__(self, target):
        domain, entity_name, question, answer = target
        r = ''

        if self.add_domain:
            r = r + domain + self.sep_token

        if self.add_entity_name:
            r = r + entity_name + self.sep_token

        r = r + question

        if self.add_answer:
            if self.separate_question_answer:
                r = r + self.sep_token
            else:
                r += ' '

            r += answer

        return r

