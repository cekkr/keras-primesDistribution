import sqlite3

class DB:
    def __init(self, path):
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()
