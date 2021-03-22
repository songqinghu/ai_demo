import pymysql

if __name__ == '__main__':
    conn = pymysql.connect(host="my-ali-03", port=3306, user="root", password="gome_search", database="groupsearch")
    cur = conn.cursor()
    cur.execute("select * from groupInfo order by id desc limit 0,5")
    item = cur.fetchone()
    print(item)
    items = cur.fetchall()
    print(items)
    conn.commit()
    cur.close()
    conn.close()
