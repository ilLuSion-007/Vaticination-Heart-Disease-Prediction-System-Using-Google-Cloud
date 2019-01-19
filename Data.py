from Database import Database as db

class Data:
    'Class to maintain user info'
    user_id = ""
    age = ""
    sex = ""
    cp = ""
    bp = ""
    sc = ""
    fs = ""
    re = ""
    mh = ""
    ig = ""
    st = ""
    stg = ""
    nv = ""
    th = ""
    def __init__(self, user_id, age, sex, cp, bp, sc, fs, re, mh, ig, st, stg, nv, th):
        self.user_id = user_id
        self.age = age
        self.sex = sex
        self.cp = cp
        self.bp = bp
        self.sc = sc
        self.fs = fs
        self.re = re
        self.mh = mh
        self.ig = ig
        self.st = st
        self.stg = stg
        self.nv = nv
        self.th = th

    def save(self):
        query = "INSERT INTO data_details (user_id, age, sex, cp, bp, sc, fs, re, mh, ig, st, stg, nv, th) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cur = db.getCursor()
        if cur.execute(query, (self.user_id, self.age, self.sex, self.cp, self.bp, self.sc, self.fs, self.re, self.mh, self.ig, self.st, self.stg, self.nv, self.th)):
           db.commit()
           return 1
        else:
            return 0

 
