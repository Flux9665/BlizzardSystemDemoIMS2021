import re
import sys
from collections import defaultdict

import numpy
import phonemizer
import torch
from cleantext import clean
from codeswitch.codeswitch import LanguageIdentification


class TextFrontend:

    def __init__(self,
                 language,
                 use_word_boundaries=False,
                 use_explicit_eos=False,
                 use_prosody=False,  # unfortunately the non segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help such systems.
                 use_lexical_stress=False,
                 use_codeswitching=False,
                 path_to_phoneme_list="PreprocessingForTTS/ipa_list.txt",
                 allow_unknown=False,
                 silent=True):
        """
        Mostly preparing ID lookups
        """
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.allow_unknown = allow_unknown
        self.use_codeswitching = use_codeswitching
        if allow_unknown:
            self.ipa_to_vector = defaultdict()
            self.default_vector = 165
        else:
            self.ipa_to_vector = dict()
        with open(path_to_phoneme_list, "r", encoding='utf8') as f:
            phonemes = f.read()
            # using https://github.com/espeak-ng/espeak-ng/blob/master/docs/phonemes.md
        phoneme_list = phonemes.split("\n")
        for index in range(1, len(phoneme_list)):
            self.ipa_to_vector[phoneme_list[index]] = index
            # note: Index 0 is unused, so it can be used for padding as is convention.
            #       Index 1 is reserved for end_of_utterance
            #       Index 2 is reserved for begin of sentence token
            #       Index 13 is used for pauses (heuristically)

        # The point of having the phonemes in a separate file is to ensure reproducibility.
        # The line of the phoneme is the ID of the phoneme, so you can have multiple such
        # files and always just supply the one during inference which you used during training.

        if language == "es":
            self.clean_lang = "es"
            self.g2p_lang = "es"
            self.expand_abbrevations = lambda x: x
            if self.use_codeswitching:
                self.lid = LanguageIdentification('spa-eng')
                self.important_en = ['gym', 'red', 'Bye', 'bye', 'Exmouth', 'Derain', 'set', 'Oxhead', 'Guy', 'VIP', 'cutre', 'confort', 'Midge', 'yen', 'USB', 'aftersun']
                with open('PreprocessingForTTS/english.city.names.txt', "r", encoding='utf8') as f:
                    self.en_cities = f.read().splitlines()
            if not silent:
                print("Created a Spanish Text-Frontend")
        else:
            print("Language not supported yet")
            sys.exit()

    def map_phones(self, phones):
        phones = phones.replace("ɔɪ", "oɪ").replace("oʊ", "o")
        phones = phones.replace("ɚɹ", "eɾ").replace("ɚ", "eɾ").replace("ɜ", "ɛɾ")
        phones = phones.replace("dʒ", "tʃ")
        phones = phones.replace("ᵻ", "i").replace("æ", "a").replace("ɔ", "o").replace("ɑ", "o").replace("ɐ", "a")
        phones = phones.replace("ə", "e").replace("ʌ", "a")

        phones = re.sub(r"(?<!a|o|e)ɪ", "i", phones)
        phones = re.sub(r"(?<!a)ʊ", "u", phones)
        phones = re.sub(r"(?<!e|ɛ)ɾ", "t", phones)

        phones = phones.replace("g", "ɣ").replace("v", "β").replace("z", "s").replace("h", "x")
        phones = phones.replace("ɹ", "ɾ").replace("ʒ", "ʃ")
        return phones

        # only use this method in case the other one doesn't work as expected

    def postprocess_codeswitch_simple(self, chunk):
        lang = chunk['lang']
        seq = chunk['word'].replace(' ##', '')
        if lang == 'es':
            seq = seq.replace(" d ' ", " d'").replace(" ' s", "'s")
        if lang == 'en-us':
            seq = seq.replace(" ' ll", "'ll").replace(" ' s", "'s").replace(" ' ve", "'ve").replace(" ' d", "'d")
        chunk['word'] = seq
        return chunk

    def postprocess_codeswitch(self, chunks):
        cleaned_chunks = [None] * len(chunks)
        ptr = 0
        for chunk in chunks:
            lang = chunk['lang']
            seq = chunk['word'].replace(' ##', '')
            # ∏print('ptr:\t', ptr, '\tseq:\t', seq, '\tlang:\t', lang)
            if lang == 'es':
                seq = seq.replace(" d ' ", " d'").replace(" ' s", "'s")
            if lang == 'en-us':
                seq = seq.replace(" ' ll", "'ll").replace(" ' s", "'s").replace(" ' ve", "'ve").replace(" ' d", "'d")
                if seq.count(" ") < 1:  # if sequence is shorter than 2 words, check if it really is english
                    if seq in self.important_en or seq in self.en_cities or "k" in seq or "w" in seq or "sh" in seq or re.search(r"^[Ss][tpk]", seq) or re.search(r"tions?$",
                                                                                                                                                                  seq) or re.search(
                        r"ings?$", seq):
                        pass
                    else:
                        lang = 'es'
            if not cleaned_chunks[ptr]:
                cleaned_chunks[ptr] = chunk
                cleaned_chunks[ptr]['word'] = seq
                cleaned_chunks[ptr]['lang'] = lang
            elif lang == cleaned_chunks[ptr]['lang']:
                cleaned_chunks[ptr]['word'] += " " + seq
            else:
                ptr += 1
                cleaned_chunks[ptr] = chunk
                cleaned_chunks[ptr]['word'] = seq
                cleaned_chunks[ptr]['lang'] = lang
        cleaned_chunks = list(filter(None, cleaned_chunks))
        return cleaned_chunks

    def string_to_tensor(self, text, view=False, return_string=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence either as IDs to be fed into an embedding
        layer, or as an articulatory matrix.
        """
        # clean unicode errors, expand abbreviations
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbrevations(utt)

        # phonemize with code switching
        if self.use_codeswitching:
            cs_dicts = self.lid.identify(utt)
            chunks = []
            for i in range(len(cs_dicts)):
                word = cs_dicts[i]['word']
                cs_lang = cs_dicts[i]['entity']
                # print(word, "\t", cs_lang)
                if cs_lang == 'spa' or cs_lang == 'other':
                    g2p_lang = 'es'
                elif cs_lang == 'en':
                    g2p_lang = 'en-us'
                elif cs_lang == 'ne':
                    if word in self.en_cities:
                        g2p_lang = 'en-us'
                    else:
                        g2p_lang = 'es'
                else:
                    g2p_lang = 'es'

                if i == 0:
                    current_lang = g2p_lang
                    current_chunk = word
                    continue

                if word.startswith('##') or word.startswith("'") or word == "s":
                    g2p_lang = current_lang  # wordpieces of one word should all have the same language

                if g2p_lang == current_lang:
                    current_chunk += " " + word
                else:
                    chunks.append({'word': current_chunk, 'lang': current_lang})
                    current_chunk = word
                    current_lang = g2p_lang
            chunks.append({'word': current_chunk, 'lang': current_lang})
            chunks = self.postprocess_codeswitch(chunks)

            # phonemize chunks
            phones_chunks = []
            for chunk in chunks:
                # chunk = self.postprocess_codeswitch_simple(chunk) # uncomment this line if postprocessing doesn't work
                seq = chunk['word']
                g2p_lang = chunk['lang']
                # print('seq: ', seq, '\t', g2p_lang)
                phones_chunk = phonemizer.phonemize(seq,
                                                    language_switch='remove-flags',
                                                    backend="espeak",
                                                    language=g2p_lang,
                                                    preserve_punctuation=True,
                                                    strip=True,
                                                    punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                                    with_stress=self.use_stress).replace(";", ",") \
                    .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                    .replace("\t", " ").replace("/", " ").replace("¡", "").replace("¿", "").replace(",", "~")

                if g2p_lang == 'en-us':
                    phones_chunk = self.map_phones(phones_chunk)
                if len(phones_chunk.split()) > 4:
                    phones_chunks.append("~" + phones_chunk + "~")
                else:
                    phones_chunks.append(phones_chunk)

            phones = ' '.join(phones_chunks)
            phones = phones.replace(" ~", "~").replace(" .", ".").replace(" !", "!").replace(" ?", "?").lstrip()
            phones = re.sub("~+", "~", phones)
        else:
            # just phonemize without code switching
            phones = phonemizer.phonemize(utt,
                                          language_switch='remove-flags',
                                          backend="espeak",
                                          language=self.g2p_lang,
                                          preserve_punctuation=True,
                                          strip=True,
                                          punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                          with_stress=self.use_stress).replace(";", ",") \
                .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                .replace("\t", " ").replace("/", " ").replace("¡", "").replace("¿", "").replace(",", "~")
            phones = re.sub("~+", "~", phones)

        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            phones = phones.replace("ˌ", "").replace("ː", "").replace("ˑ", "").replace("˘", "").replace("|", "").replace("‖", "")

        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        else:
            phones = re.sub(r"\s+", " ", phones)

        phones = "+" + phones

        # I have no idea how this happened, but the synthesis just cannot pronounce ɔ.
        # Seems like it did not occur in the training data, maybe aligner removed it? As hacky fix, use o instead.
        phones = phones.replace("ɔ", "o") + "~"
        # phones = self.map_phones(phones)

        if view:
            print("Phonemes: \n{}\n".format(phones))

        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if self.allow_unknown:
                phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
            else:
                if char in self.ipa_to_vector.keys():
                    phones_vector.append(self.ipa_to_vector[char])
        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])

        # combine tensors and return
        if not return_string:
            return torch.LongTensor(phones_vector).unsqueeze(0)
        else:
            return phones

    def phones_to_tensor(self, phones):
        phones = phones.replace("_p:_", "~")
        phones_with_dur = phones.split(" ")
        phones = ""
        for phone_with_dur in phones_with_dur:
            phones += phone_with_dur.split("_")[0]
        phones = "+" + phones.rstrip("~").lstrip("~")  # the EOS will be added by the synthesis to ensure that there is always one there
        phones_vector = list()
        for char in phones:
            try:
                phones_vector.append(self.ipa_to_vector[char])
            except KeyError:
                print("Unknown symbol produced by the aligner: {}".format(char))
        return torch.LongTensor(phones_vector).unsqueeze(0)

    def phones_and_dur_to_tensor(self, phones, melspec):
        phones = phones.replace("_p:_", "~")
        if not phones[0] == "~":
            phones = "~_0.0001 " + phones
        if not phones.split()[-1][0] == "~":
            phones = phones + " ~_0.0001"
        phones_with_dur = phones.split(" ")
        phones = ""
        durs = []
        for phone_with_dur in phones_with_dur:
            phone, _, dur = phone_with_dur.partition("_")
            phones += phone
            durs.append(float(dur))
        phones = "+" + phones.rstrip("~").lstrip("~")  # the EOS will be added by the synthesis to ensure that there is always one there
        phones_vector = list()
        for char in phones:
            try:
                phones_vector.append(self.ipa_to_vector[char])
            except KeyError:
                print("Unknown symbol produced by the aligner: {}".format(char))

        x = len(melspec) / sum(durs)
        frames = [round(dur * x) for dur in durs]
        diff = len(melspec) - sum(frames)

        # if number of frames doesn't match, adjust long durations first
        if diff > 0:
            sorted_frames = numpy.argsort(frames)[::-1]
            for i in range(diff):
                idx = sorted_frames[i]
                frames[idx] += 1
        elif diff < 0:
            sorted_frames = numpy.argsort(frames)[::-1]
            for i in range(abs(diff)):
                idx = sorted_frames[i]
                frames[idx] -= 1
        assert sum(frames) == len(melspec), "Number of calculated frames does not match number of spectrogram frames"

        return torch.LongTensor(phones_vector).unsqueeze(0), torch.LongTensor(frames)


if __name__ == '__main__':
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # test a Spanish utterance
    tfr_de = TextFrontend(language="es", use_word_boundaries=True, use_explicit_eos=False, use_codeswitching=False,
                          path_to_phoneme_list="ipa_list.txt")
    print(tfr_de.string_to_tensor(
        "Es una película de robots donde trabaja una niña que se llama Megan Fox.",
        view=True))

    # test a Spanish utterance with code switching
    tfr_de = TextFrontend(language="es", use_word_boundaries=True, use_explicit_eos=False, use_codeswitching=True,
                          path_to_phoneme_list="ipa_list.txt")
    print(tfr_de.string_to_tensor(
        "Es una película de robots donde trabaja una niña que se llama Megan Fox.",
        view=True))
