import MySQLdb as mysql

class Database:
    'Class for database connection to mysql'
    db = None

    @staticmethod
    def getCursor(server = '35.196.146.215', port=3306, user ='upestech', password ='vaticination', database = 'vaticination'):
        if Database.db is None:
            Database.db = mysql.connect("35.196.146.215", "upestech", "vaticination", "vaticination")
        return Database.db.cursor()

    @staticmethod
    def commit():
        Database.db.commit()

    