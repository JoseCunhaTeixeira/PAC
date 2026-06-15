from masw.io.paths import INPUT_DIR


def get_input_folders() -> list[str]:

    print(f"INPUT_DIR = {INPUT_DIR}")

    if not INPUT_DIR.exists():
        print("INPUT_DIR does not exist")
        return []

    folders = sorted(folder.name for folder in INPUT_DIR.iterdir() if folder.is_dir())

    print(f"Folders found: {folders}")

    return folders
