import sqlite3

path = 'templates/db3.db'
conn = sqlite3.connect(path)

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    c = conn.cursor()
    c.execute(create_table_sql)


userRegTable = """ CREATE TABLE IF NOT EXISTS projects (
                                        personid integer PRIMARY KEY,
                                        name text NOT NULL,
                                        password text NOT NULL,
                                        email text NOT NULL
                           ); """
create_table(conn,userRegTable)
conn.commit()
conn.close()