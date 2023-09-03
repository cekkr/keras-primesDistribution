import sqlite3

class DB:
    def __init(self, path):
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()
        self.tables = {}

    def exists(self, name):
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='"+name+"'"
        res = self.cur.execute(sql).fetchone()
        return res is not None

    def get(self, name):
        if self.tables[name] is None:
            self.tables[name] = Table(self, name)

        return self.tables[name]

class Table:
    def __init(self, db, name):
        self.db = db
        self.cols = {}

    def __setattr__(self, name, value):
        self.cols[name] = value

    def _create(self):
        self.id = 'INTEGER NOT NULL PRIMARY KEY'
        #todo: create table

    def insert(self, **kwargs):
        for k,v in kwargs.items():
            print(k,v)