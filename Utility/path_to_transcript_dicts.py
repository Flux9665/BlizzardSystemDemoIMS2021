import os


def build_path_to_transcript_dict_spanish_big_train():
    root = "/mount/resources/speech/corpora/Spanish-100hrs/tux-100h/valid"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata_phones.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[2]
            wav_path = os.path.join(root, "wavs", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_spanish_blizzard_train():
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    path_to_transcript = dict()
    with open(os.path.join(root, "train_phones.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript  # [flx] removed .lower() to save acronyms for the phonemizer
    return path_to_transcript
