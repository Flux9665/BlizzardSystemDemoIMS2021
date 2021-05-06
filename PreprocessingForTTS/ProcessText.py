import os
import re
import sys
from collections import defaultdict

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
                 path_to_panphon_table="PreprocessingForTTS/ipa_vector_lookup.csv",
                 silent=True):
        """
        Mostly preparing ID lookups
        """
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.use_codeswitching = use_codeswitching
        self.ipa_to_vector = defaultdict()
        self.default_vector = 133
        with open(path_to_panphon_table, encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            self.ipa_to_vector[line_list[0]] = index
            # note: Index 0 is unused, so it can be used for padding as is convention.
            #       Index 1 is reserved for EOS, if you want to use explicit EOS.
            #       Index 133 is used for unknown characters
            #       Index 12 is used for pauses (heuristically)
            #       Index 132 is used for begin_of_sentence
        if language == "es":
            self.clean_lang = "es"
            self.g2p_lang = "es"
            self.expand_abbrevations = lambda x: x
            self.lid = LanguageIdentification('spa-eng')
            if not silent:
                print("Created a Spanish Text-Frontend")
        else:
            print("Language not supported yet")
            sys.exit()

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

        # if an aligner has produced silence tokens before, turn
        # them into silence markers now so that they survive the
        # phonemizer:
        utt = utt.replace("_SIL_", "~")

        # code switching
        if self.use_codeswitching:

            cs_dicts = self.lid.identify(utt)

            # convert wordpiece tokens back to words
            id_to_del = []
            for i in range(len(cs_dicts)):
                word = cs_dicts[i]['word']
                if word.startswith('##'):
                    id_to_del.append(i)
                    cs_dicts[i - 1]['word'] += word.replace('##', '')
                else:
                    continue

            for idx in sorted(id_to_del, reverse=True):
                del cs_dicts[idx]
            # print('results after deletions: ')

            phones = ""
            for entry in cs_dicts:
                word = entry['word']
                cs_lang = entry['entity']
                if cs_lang == 'spa':
                    g2p_lang = 'es'
                elif cs_lang == 'en':
                    g2p_lang = 'en-us'
                else:
                    g2p_lang = 'en-us'  # try en as default, since most Named Entities in dev set should be pronounced in English

                # phonemize word by word
                phones += phonemizer.phonemize(word,
                                               language_switch='remove-flags',
                                               backend="espeak",
                                               language=g2p_lang,
                                               preserve_punctuation=True,
                                               strip=True,
                                               punctuation_marks=';:,.!?¡¿—…"«»“”~',
                                               with_stress=self.use_stress).replace(";", ",") \
                    .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                    .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")
            phones = re.sub("~+", "~", phones)

        else:
            # phonemize
            phones = phonemizer.phonemize(utt,
                                          language_switch='remove-flags',
                                          backend="espeak",
                                          language=self.g2p_lang,
                                          preserve_punctuation=True,
                                          strip=True,
                                          punctuation_marks=';:,.!?¡¿—…"«»“”~',
                                          with_stress=self.use_stress).replace(";", ",") \
                .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")
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

        if view:
            print("Phonemes: \n{}\n".format(phones))

        tensors = list()
        phones_vector = list()

        # turn into numeric vectors
        for char in phones:
            phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))

        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])

        # turn into tensors
        tensors.append(torch.LongTensor(phones_vector))

        # combine tensors and return
        if not return_string:
            return torch.stack(tensors, 0)
        else:
            return phones + "#"

    def phones_to_tensor(self, phones):
        phones = phones.replace("_SIL_", "~")
        phones = phones.replace(" ", "")
        phones = "+" + phones.rstrip("~").lstrip("~")  # the EOS will be added by the synthesis to ensure that there is always one there
        phones_vector = list()
        for char in phones:
            phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
        return torch.LongTensor(phones_vector).unsqueeze(0)


def phonemize_train_text_no_silences_no_eos_blizzard():
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    tf = TextFrontend(language="es", use_word_boundaries=True)
    path_to_transcript = dict()
    with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            file_handle = line.split("\t")[0]
            path_to_transcript[file_handle] = tf.string_to_tensor(norm_transcript, view=False, return_string=True)
    resulting_file_content = ""
    for key in path_to_transcript:
        resulting_file_content += key + "\t" + path_to_transcript[key] + "\n"
    resulting_file_content = resulting_file_content[:-2]  # remove the final linebreak
    with open(os.path.join(root, "train_phones.txt"), "w", encoding="utf8") as file:
        file.write(resulting_file_content.replace("~", "").replace("+", ""))
    print("Done with Blizzard")


def phonemize_train_text_no_silences_no_eos_big():
    root = "/mount/resources/speech/corpora/Spanish-100hrs/tux-100h/valid"
    tf = TextFrontend(language="es", use_word_boundaries=True)
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[2]
            file_handle = line.split("|")[0]
            path_to_transcript[file_handle] = tf.string_to_tensor(norm_transcript, view=False, return_string=True)
            resulting_file_content = ""
            for key in path_to_transcript:
                resulting_file_content += key + "\t" + path_to_transcript[key] + "\n"
            resulting_file_content = resulting_file_content[:-2]  # remove the final linebreak
            with open(os.path.join(root, "train_phones.txt"), "w", encoding="utf8") as file:
                file.write(resulting_file_content.replace("~", "").replace("+", ""))
    print("Done with Big")


if __name__ == '__main__':
    # test a Spanish utterance
    tfr_de = TextFrontend(language="es", use_panphon_vectors=False, use_word_boundaries=False, use_explicit_eos=False,
                          path_to_panphon_table="ipa_vector_lookup.csv")
    print(tfr_de.string_to_tensor("Desde los sesudos editoriales de 'The Washington Post' al Palacio del Elseo.", view=True))

    # test a Spanish utterance with code switching
    tfr_de = TextFrontend(language="es", use_panphon_vectors=False, use_word_boundaries=False, use_explicit_eos=False, use_codeswitching=True,
                          path_to_panphon_table="ipa_vector_lookup.csv")
    print(tfr_de.string_to_tensor("Desde los sesudos editoriales de 'The Washington Post' al Palacio del Elseo.", view=True))
