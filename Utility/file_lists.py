from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_spanish_blizzard_train


def get_file_list_spanish_blizzard():
    return list(build_path_to_transcript_dict_spanish_blizzard_train().keys())
