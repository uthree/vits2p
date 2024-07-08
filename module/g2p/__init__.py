from .japanese import JapaneseModule
import torch


class G2PModule:
    def __init__(self):
        self.language_modules = dict()

        self.language_modules['ja'] = JapaneseModule()

        self.phonemes = ['<pad>']
        for m in self.language_modules.values():
            self.phonemes += (m.vocabulary())

    def grapheme_to_phoneme(self, text, language):
        module = self.language_modules[language]
        phonemes = module.g2p(text)
        return phonemes
    
    def phonemes_to_ids(self, phoneme_sequence):
        output = []
        for p in phoneme_sequence:
            id = self.phonemes.index(p)
            output.append(id)
        return output
    
    def get_language_id(self, language):
        return self.language_modules.keys().index(language)
    
    def encode(self, text, language, max_length=None):
        '''
        Outputs:
            phoneme_ids: shape=[1, L]
            lengths: shape[1]
            language_ids: shape=[1]
        '''
        phonemes = self.grapheme_to_phoneme(text, language)
        length = len(phonemes)
        ids = self.phonemes_to_ids(phonemes)
        if max_length is not None:
            while len(ids) < max_length:
                ids.append(0) # add padding
            if len(ids) > max_length:
                ids = ids[:max_length]
        language_id = self.languages().index(language)
        return torch.LongTensor(ids).unsqueeze(0), torch.LongTensor([length]), torch.LongTensor([language_id])
    
    def languages(self):
        return list(self.language_modules.keys())