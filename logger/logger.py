import datetime

class Logger:
    def __init__(self, file_name):
        self.file_name = file_name

    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.file_name, 'a') as file:
            file.write(f'{timestamp} - {message}\n')


# Usage example:
logger = Logger('log.txt')
logger.log('This is a log message.')
logger.log('This is another log message.')
