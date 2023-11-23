import os
import random
import re
import shutil


def read_deepfake_not_cleaned(root_fake):
    deepfake_filenames = list(filter(lambda x: x.endswith(".flac"), os.listdir(root_fake)))
    re_pattern = r"([0-9]+)-([0-9]+)"
    voices = {}
    for deepfake_file in deepfake_filenames:
        re_match = re.match(re_pattern, deepfake_file)
        if not re_match:
            print("MATCH NOT FOUND ERROR")
            continue
        voice = re_match.group(2)
        if voice not in voices.keys():
            voices[voice] = [deepfake_file]
        else:
            voices[voice].append(deepfake_file)
    return voices


def read_real_voices_not_cleaned(root_real):
    re_pattern = r"([0-9]+)-([0-9]+)-([0-9]+)"
    voices = {}
    for folder in os.listdir(root_real):
        path_to_folder = os.path.join(root_real, folder)
        for subfolder in os.listdir(path_to_folder):
            path_to_subfolder = os.path.join(path_to_folder, subfolder)
            real_filenames = list(filter(lambda x: x.endswith(".flac"), os.listdir(path_to_subfolder)))
            for filename in real_filenames:
                re_match = re.match(re_pattern, filename)
                if not re_match:
                    print("MATCH NOT FOUND ERROR")
                    continue
                voice = re_match.group(2)
                line = re_match.group(3)
                if voice not in voices.keys():
                    voices[voice] = {line: os.path.join(path_to_subfolder, filename)}
                else:
                    voices[voice][line] = os.path.join(path_to_subfolder, filename)
    return voices


def get_splitted_voice_filenames(voices, train_data_percentage):
    voice_keys = list(voices.keys())
    random.shuffle(voice_keys)
    train_voices = voice_keys[:int(train_data_percentage * len(voice_keys))]
    test_voices = voice_keys[int(train_data_percentage * len(voice_keys)):]

    train_filenames = []
    for voice in train_voices:
        train_filenames.extend(voices[voice])

    test_filenames = []
    for voice in test_voices:
        test_filenames.extend(voices[voice])

    return train_filenames, test_filenames


def copy_real_deepfake(deepfake_filelist, real_voice_dict, destination_path_fake,
                       destination_path_real, root_fake):
    for i, deepfake_filename in enumerate(deepfake_filelist):
        print(f"{round(100 * (i + 1) / len(deepfake_filelist), 2)}%, {deepfake_filename}")
        pattern = r"([0-9]+)-([0-9]+)"
        match = re.match(pattern, deepfake_filename)
        if not match:
            print("MATCH NOT FOUND ERROR")
            continue
        line_idx = match.group(1)
        voice_idx = match.group(2)
        src_deepfake_path = os.path.join(root_fake, deepfake_filename)
        dst_deepfake_path = os.path.join(destination_path_fake, f"{voice_idx}-0{line_idx}.flac")

        src_real_voice_path = real_voice_dict[voice_idx][f"0{line_idx}"]
        dst_real_voice_path = os.path.join(destination_path_real, f"{voice_idx}-0{line_idx}.flac")

        shutil.copyfile(src_real_voice_path, dst_real_voice_path)
        shutil.copyfile(src_deepfake_path, dst_deepfake_path)


def create_jcorentin_dataset(real_root, fake_root, dst_root="./resources", test_train_ratio=0.8):
    destination_train_real = os.path.join(dst_root, "train", "real")
    destination_train_fake = os.path.join(dst_root, "train", "fake")
    destination_valid_real = os.path.join(dst_root, "valid", "real")
    destination_valid_fake = os.path.join(dst_root, "valid", "fake")
    for path in [destination_train_real, destination_train_fake, destination_valid_real, destination_valid_fake]:
        if not os.path.exists(path):
            os.makedirs(path)

    real_voices_dict = read_real_voices_not_cleaned(real_root)
    fake_voices_dict = read_deepfake_not_cleaned(fake_root)
    train_f, test_f = get_splitted_voice_filenames(fake_voices_dict, test_train_ratio)

    copy_real_deepfake(train_f, real_voices_dict, destination_train_fake, destination_train_real, fake_root)
    copy_real_deepfake(test_f, real_voices_dict, destination_valid_fake, destination_valid_real, fake_root)


def copy_files_to_destination_11labs(destination_real, destination_fake, voices_to_copy, voices_dict):
    files_to_copy_length = 0
    processed_files = 0
    for voice in voices_to_copy:
        files_to_copy_length += len(list(voices_dict[voice].keys()))
    for i, voice in enumerate(voices_to_copy):
        for j, (line, (src_real_voice_path, src_fake_voice_path)) in enumerate(voices_dict[voice].items()):
            print(f"{round(100 * (processed_files + 1) / files_to_copy_length, 2)}%, {voice}-{line}")
            shutil.copyfile(src_real_voice_path, os.path.join(destination_real, f"{voice}-{line}.flac"))
            shutil.copyfile(src_fake_voice_path, os.path.join(destination_fake, f"{voice}-{line}.flac"))
            processed_files += 1


def create_11labs_dataset(real_root, fake_root, dst_root="./elevenlabs", test_train_ratio=0.8):
    # 1. Creating folders
    destination_train_real = os.path.join(dst_root, "train", "real")
    destination_train_fake = os.path.join(dst_root, "train", "fake")
    destination_valid_real = os.path.join(dst_root, "valid", "real")
    destination_valid_fake = os.path.join(dst_root, "valid", "fake")
    for path in [destination_train_real, destination_train_fake, destination_valid_real, destination_valid_fake]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 2. Aggregating files to dict with structure:
    # voice_index 12320:
    #   line_index 0000:
    #       (path_to_real_file, path_to_fake_file)
    #   line_index 0001:
    #       (path_to_real_file, path_to_fake_file)
    #   line_index 0002:
    #       (path_to_real_file, path_to_fake_file)
    re_pattern = r"([0-9]+)-([0-9]+)-([0-9]+)"
    voices = {}
    for folder in os.listdir(fake_root):
        path_to_folder = os.path.join(fake_root, folder)
        if not os.path.isdir(path_to_folder):
            continue
        for subfolder in os.listdir(path_to_folder):
            path_to_subfolder = os.path.join(path_to_folder, subfolder)
            if not os.path.isdir(path_to_subfolder):
                continue
            fake_filenames = list(filter(lambda x: x.endswith(".mp3"), os.listdir(path_to_subfolder)))
            for filename in fake_filenames:
                re_match = re.match(re_pattern, filename)
                if not re_match:
                    print("MATCH NOT FOUND ERROR")
                    continue
                voice = re_match.group(2)
                line = re_match.group(3)
                src_path_to_fake_file = os.path.join(path_to_subfolder, filename)
                src_path_to_real_file = os.path.join(real_root, folder, subfolder, filename[:-16] + ".flac")
                if voice not in voices.keys():
                    voices[voice] = {line: (src_path_to_real_file, src_path_to_fake_file)}
                else:
                    voices[voice][line] = (src_path_to_real_file, src_path_to_fake_file)

    # 3. Splitting into train and test sets. All the lines from one voice will go to the same set
    voice_keys = list(voices.keys())
    random.shuffle(voice_keys)
    train_voices = voice_keys[:int(test_train_ratio * len(voice_keys))]
    test_voices = voice_keys[int(test_train_ratio * len(voice_keys)):]

    # 4. Copying files to destination
    copy_files_to_destination_11labs(destination_train_real, destination_train_fake, train_voices, voices)
    copy_files_to_destination_11labs(destination_valid_real, destination_valid_fake, test_voices, voices)







if __name__ == "__main__":
    """
    1. Download jcorentin deepfake dataset from https://projektbadawczystorage.blob.core.windows.net/deepfake-audio-dataset/ready-dataset-packages/LibriSpeechDeepfake.zip
    2. Download real voices from (train-clean-100.tar.gz [6.3G]) https://www.openslr.org/12 
    3. Set root_real and root_fake to correct folders
    4. Run the script (It can take a while)
    """
    # root_real = r"C:\Users\CamaroTheBOSS\Downloads\LibriSpeech\train-clean-100"
    # root_fake = r"C:\Code\Real-Time-Voice-Cloning\datasets\LibriSpeechDeepfake"
    # create_jcorentin_dataset(root_real, root_fake)

    """
    1. Download 11labs deepfake dataset from https://projektbadawczystorage.blob.core.windows.net/deepfake-audio-dataset/ready-dataset-packages/deepfakes_11labs.zip
    2. Download real voices from (test-clean.tar.gz [346M]) https://www.openslr.org/12 
    3. Set root_real and root_fake to correct folders
    4. Run the script (It can take a while)
    """
    root_real = r"C:\Code\Real-Time-Voice-Cloning\LibriSpeech\dev-clean"
    root_fake = r"C:\Users\CamaroTheBOSS\OneDrive\Desktop\deepfakes_11labs"
    create_11labs_dataset(root_real, root_fake)


