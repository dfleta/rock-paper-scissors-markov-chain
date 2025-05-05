import os


class UserActionsFileHandler:
    FILE_PATH = "./user_actions_history.txt"

    @classmethod
    def read_actions(cls):
        try:
            with open(cls.FILE_PATH, "r") as file:
                return [
                    list(map(int, line.strip("[]\n").split(", ")))
                    for line in file
                    if line.strip()
                ]
        except FileNotFoundError:
            return []

    @classmethod
    def append_actions(cls, actions):
        os.makedirs(os.path.dirname(cls.FILE_PATH), exist_ok=True)
        with open(cls.FILE_PATH, "a") as file:
            file.write(repr(actions) + "\n")
