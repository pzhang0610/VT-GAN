import logging
import os
import datetime
import shutil

class myLog(object):

    def __init__(self, log_base_dir):
        self.log_base_dir = log_base_dir
        self.log_name = __name__
        self.logger, self.log_dir = self._init_log()

    def _init_log(self):
        if self.log_base_dir is None:
            self.log_base_dir = os.getcwd()
        if not os.path.exists(self.log_base_dir):
            os.makedirs(self.log_base_dir)
        else:
            [shutil.rmtree(os.path.join(self.log_base_dir, i)) for i in os.listdir(self.log_base_dir)]
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_dir = os.path.join(self.log_base_dir, now)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # init logging
        logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(self.log_name)

        logger.setLevel(logging.DEBUG)
        return logger, log_dir

    def add_log(self, log_info, is_print=False):
        self.logger.info(log_info)
        if is_print:
            print(log_info)
