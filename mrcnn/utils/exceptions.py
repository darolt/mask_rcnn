

class NoBoxHasPositiveArea(Exception):
    def __init__(self):
        Exception.__init__(self, 'No box has positive area.')


class NoBoxToKeep(Exception):
    def __init__(self):
        Exception.__init__(self, 'No box to keep in detection.')
