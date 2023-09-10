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

    def __getattr__(self, item):
        return self.get(item)

class Table:
    def __init(self, db, name):
        self.db = db
        self.name = name
        self.cols = {}

    def __setattr__(self, name, value):
        if len(self.cols) == 0:
            self.cols['id'] = 'INTEGER NOT NULL PRIMARY KEY'

        self.cols[name] = value

    def create(self):
        query = 'CREATE TABLE ' + self.name + ' ('

        for k, v in self.cols.items():
            query += k + ' ' + v

        query += ')'

        self.db.cur.execute(query)

    def insert(self, **kwargs):
        tup = ()
        cols = ""
        what = ""

        for k,v in kwargs.items():
            if len(cols) > 0:
                cols += ", "
                what += ", "

            cols += k
            what += "?"

            tup.append(v)

        self.db.cur.execute("INSERT INTO "+self.name+" ("+cols+") VALUES ("+what+")", tup)
        self.db.cur.commit()

    def query(self):
        return QueryBuilder(self.db, self.name)

class QueryBuilder:
    def __init__(self, db, table=None):
        self.db = db
        self.table = table
        self.where = QueryBuilderSetter()
        self.quests = []

    @property
    def type(self):
        return self.type

    @type.setter
    def type(self, val):
        self.type = val

    @property
    def what(self):
        return self.what

    @what.setter
    def what(self, val):
        self.what = val

    @property
    def where(self):
        return self.where

    @where.setter
    def where(self, val):
        self.where = val

    def build(self):
        q = ''
        if self.type == 'SELECT':
            q += self.type + ' '
            q += self.what + ' '

            # WHERE
            q += 'WHERE '
            for k in self.where.ord:
                if k in self.where.vals:
                    q += k + '? '
                    self.quests.append(self.where.vals[k])
                else:
                    q += k


class QueryBuilderSetter:
    def __init__(self):
        self.vals = {}
        self.ord = []

    def __setattr__(self, item, val):
        self.vals[item] = val
        self.ord.append(item)

    def then(self, then):
        self.ord.append(then)
